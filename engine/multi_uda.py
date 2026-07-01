import random
from typing import List, Dict, Tuple, Union, Any, Optional
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from ema_pytorch import EMA
from transformers.modeling_outputs import BaseModelOutput
from scipy.optimize import linear_sum_assignment
import torch.nn as nn

from engine.dataloader import RareCategoryManager
from engine.category import Category

class FocalLoss(torch.nn.Module):
    """
    數值穩定版 Focal Loss。

    改用 PyTorch 內建 F.cross_entropy（C++ log-sum-exp kernel）計算每像素 CE loss，
    再以 pt = exp(-CE) 反推各像素被正確分類的機率，最後乘上 focal weight (1-pt)^gamma 與 alpha。

    數學等價性：
        原版: loss = -(1-pt)^γ · α · log(pt)
        新版: CE = -log(pt)，pt = exp(-CE)
              loss = α · (1-pt)^γ · CE  ← 完全等價

    F.cross_entropy 在內部做 log-sum-exp 不會有 FP16 underflow → -inf 問題，
    pt = exp(-CE) 的值域永遠是 [0, 1]，不會 overflow。
    """
    def __init__(self, gamma=0, alpha=None, reduction=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # 強制 FP32：避免 FP16 autocast 下 log_softmax underflow → -inf
        input = input.float()

        # F.cross_entropy 需要 long 型 target
        target = target.long()

        # ignore_index=None 時沿用 PyTorch 預設值 -100（不影響任何正常 label）
        ignore_idx = self.ignore_index if self.ignore_index is not None else -100

        # 每像素 CE loss（C++ kernel，數值穩定）
        # 回傳 shape 與 target 相同，如 [B, H, W] 或 [B]
        # ignore 位置的 ce_loss 由 PyTorch 保證為 0.0
        ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=ignore_idx)

        # pt = p(正確類別) = exp(-CE)，值域 [0, 1]，detach 避免梯度穿透 focal weight
        pt = torch.exp(-ce_loss.detach())

        # focal weight：(1 - pt)^gamma
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(device=input.device, dtype=input.dtype)
            # 建立安全 target 以供 gather：將 ignore 位置替換為 0 避免 out-of-bounds
            # 這些位置的 focal_loss 最終會被歸零，所以 alpha 值不影響結果
            target_safe = target.clone()
            if self.ignore_index is not None:
                target_safe = target_safe.masked_fill(target == self.ignore_index, 0)
            alpha_weight = alpha_t[target_safe]   # 與 target 同 shape
            focal_loss = alpha_weight * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        # 顯式將 ignore 位置歸零
        # （F.cross_entropy 已保證 ce_loss=0 → focal_loss 理論上也是 0，
        #  但 alpha_weight 可能非零，這一行保證計算結果乾淨）
        if self.ignore_index is not None:
            focal_loss = focal_loss.masked_fill(target == self.ignore_index, 0.0)

        if self.reduction == 'mean':
            if self.ignore_index is not None:
                valid_count = (target != self.ignore_index).sum()
                if valid_count == 0:
                    return torch.tensor(0.0, device=input.device, requires_grad=True)
                return focal_loss.sum() / valid_count
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            # reduction=None：回傳每像素 loss，供 PixelThreshold 用布林遮罩取樣
            return focal_loss

class PixelThreshold:
    """
    通用閾值過濾與 Loss 計算模組。
    不需要修改，因為它只處理 Logits 和 Annotations，不依賴模型架構。
    """
    def __init__(self, threshold: float = 0.968, focal_loss: FocalLoss = None) -> None:
        assert isinstance(threshold, float)
        self.threshold = threshold
        self.focal_loss = focal_loss

    def compute(
        self,
        logits: torch.Tensor,
        soft_ann: torch.Tensor,
        class_weights: torch.Tensor = None,
        class_thresholds: list = None,  # per-class threshold list, len == num_classes
    ) -> torch.Tensor:
        prob, ann = soft_ann.max(1)

        if self.focal_loss is not None:
            loss = self.focal_loss.forward(input=logits, target=ann)
        elif logits.shape[1] > 1:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=class_weights,
                ignore_index=255,
                reduction='none'
            )
            loss = loss_fct(logits, ann)
        elif logits.shape[1] == 1:
            valid_mask = ((ann >= 0) & (ann != 255)).float()
            loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
            loss = loss_fct(logits.squeeze(1), ann.float())
            loss = (loss * valid_mask)

        if class_thresholds is not None:
            # 每個 pixel 的門檻依 teacher 預測的類別而定
            thresh_map = torch.tensor(class_thresholds,
                                      dtype=prob.dtype, device=prob.device)[ann]
            ge = prob >= thresh_map
        else:
            ge = prob >= self.threshold

        if ge.any():
            return loss[ge].mean()
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

class MultiTaskClassMix:
    """
    適配多任務模型的 ClassMix 數據增強模組。
    """
    def __init__(
        self,
        device: torch.device,
        batch_size: int,
        rcm: RareCategoryManager,
        categories: List[Category],
        source_dataloader: DataLoader = None,
        target_dataloader: DataLoader = None,
        ema: EMA = None,
        mix_num: int = 1
    ) -> None:
        self.device = device
        self.batch_size = batch_size
        self.rcm = rcm
        self.categories = categories
        self.source_dataloader = source_dataloader
        self.target_dataloader = target_dataloader
        self.ema = ema
        self.mix_num = mix_num

    def mix(
        self,
        mix_domain: int,
        imgs: List[torch.Tensor],
        ann: torch.Tensor,
        task: str, # [New] 必須指定任務 ('rm' or 'da') 以決定使用哪個 Teacher Head
        dis_label: torch.Tensor = None,
        erased_imgs: List[torch.Tensor] = None,
        class_thresholds: list = None,  # 類別特定偽標籤閾值，與 PixelThreshold.compute 保持一致
    ):
        """
        Args:
            mix_domain: 0 for source, 1 for target
            imgs: List of image tensors (current batch)
            ann: Annotation tensor (current batch)
            task: 'rm' or 'da', specifies which decoder to use for pseudo-labeling
            ...
        """
        # 1. 獲取 Sample Data (從 Source 或 Target DataLoader)
        sample_data = next(self.source_dataloader if mix_domain == 0 else self.target_dataloader)
        sample_imgs = sample_data["imgs"] if "imgs" in sample_data else [sample_data["img"]]
        sample_imgs = [im.to(self.device) for im in sample_imgs]
        
        # 2. Teacher (EMA) 模型推論
        # [Modify] 呼叫 EMA 時傳入 task 參數，並處理 Dict 回傳格式
        # MultiTaskSegformer 回傳 {'logits': ..., 'loss': ...}
        # BN 在 training mode 下使用當前 batch 統計，batch_size=1 時極不穩定 → 偽標籤噪音大
        # eval mode 使用 running stats（穩定），推論結束後切回 train 避免影響後續 EMA update
        self.ema.ema_model.eval()
        with torch.no_grad():
            # [Fix] sample_imgs 是一個 list [Tensor]，但模型只接受 Tensor。
            # 我們取出第一張圖 (index 0) 傳入模型。
            # 如果是 EMA wrapper，通常 forward 到 model(pixel_values, ...)
            input_tensor = sample_imgs[0]

            output = self.ema(pixel_values=input_tensor, task=task)
        self.ema.ema_model.train()
            
        # 根據回傳結構取得 Logits (假設 EMA 正確轉發了 kwargs)
        if isinstance(output, dict):
            sample_logits = output["logits"]
        else:
            # Fallback 若 output 僅是 logits (視 EMA 實作而定)
            sample_logits = output[0] if isinstance(output, tuple) else output

        sample_ann = sample_logits.softmax(1)
        
        # 將 Sample 移回 CPU 進行 mask 處理
        sample_imgs = [im.to(torch.device('cpu')) for im in sample_imgs]
        sample_ann = sample_ann.to(torch.device('cpu'))

        # 3. 準備 Hard Labels (用於混合)
        if mix_domain == 0:
            # 如果是 Source domain，直接用 Ground Truth
            hard_ann = sample_data["ann"].squeeze(1)   # [H, W]
        else:
            # 如果是 Target domain，使用 Teacher 的 Pseudo-label
            ann_prob, hard_ann = sample_ann.max(1)
            if class_thresholds is not None:
                thresh_map = torch.tensor(class_thresholds, dtype=ann_prob.dtype, device=ann_prob.device)[hard_ann]
                hard_ann[ann_prob < thresh_map] = 0
            else:
                hard_ann[ann_prob < 0.968] = 0

        # 4. ClassMix 核心邏輯 (Masking & Paste)
        for idx in range(0, self.batch_size):
            mix_cat_list = self.rcm.get_mix_cat_id(cateList=torch.unique(hard_ann).tolist(), mix_num=self.mix_num)
            
            for mix_cat in mix_cat_list:
                if mix_cat != 0:
                    mix_idx = next((i for i in range(0, hard_ann.shape[0]) if mix_cat in hard_ann[i]), None)
                    if mix_idx is None: continue

                    # mix_domain 直接決定 patch 的 domain 歸屬（0=source, 1=target）
                    # 不再讀 sample_data["domain"]（天氣標籤，語義不等於 source/target membership）
                    mix_domain_val = mix_domain
                    
                    mix_mask = hard_ann[mix_idx].clone()
                    mix_mask[mix_mask != mix_cat] = 0
                    
                    # [新增] 準備不同用途的 Mask
                    mix_mask4imgs = mix_mask.unsqueeze(0).repeat(3, 1, 1)                  # RGB 圖片用 (3, H, W)
                    mix_mask4soft_ann = mix_mask.unsqueeze(0).repeat(len(self.categories), 1, 1) # Soft Labels 用 (C, H, W)
                    mix_mask4dis = mix_mask.unsqueeze(0)                                   # [Fix] Domain Label 用 (1, H, W)
                    
                    mix_imgs_data = [im[mix_idx] for im in sample_imgs]
                    mix_ann_data = sample_ann[mix_idx]

                    # 應用混合到當前 batch (imgs, ann)
                    # 注意：imgs 也是 list of tensors，這裡我們對每一個版本都做 mix
                    for i in range(0, len(imgs)):
                        if i < len(mix_imgs_data):
                            imgs[i][idx][mix_mask4imgs != 0] = mix_imgs_data[i][mix_mask4imgs != 0]
                    
                    ann[idx][mix_mask4soft_ann != 0] = mix_ann_data[mix_mask4soft_ann != 0]
                    
                    if dis_label is not None:
                        # [Fix] 使用 mix_mask4dis (1通道) 來索引 dis_label (1通道)
                        dis_label[idx][mix_mask4dis != 0] = mix_domain_val
            
            # 處理 Erased Images (如果有)
            if erased_imgs is not None:
                for i in range(0, len(imgs)):
                    if i < len(erased_imgs):
                        erased_mask = (erased_imgs[i][idx] != 0).all(dim=0)
                        erased_mask = erased_mask.unsqueeze(0).repeat(3, 1, 1)
                        erased_imgs[i][idx][erased_mask] = imgs[i][idx][erased_mask]

def compute_domain_discrimination_loss_mt(
    model: torch.nn.Module, # 這裡預期是 MultiTaskSegformer
    imgs: List[torch.Tensor],
    ann: torch.Tensor,
    dis_label: torch.Tensor,
    domain_class_weight: torch.Tensor,
    crop_size: Tuple[int, int], 
    task: str, # [New] 指定任務 ('rm' or 'da') 以選擇對應的 Discriminator
    device: torch.device,
    spatial_mask: torch.Tensor = None,  # [NEW] 新增物件感知遮罩參數
    latent: tuple = None  # [NEW] 接收已計算好的隱藏特徵
) -> torch.Tensor:
    """
    計算多任務模型的領域判別損失 (Domain Discrimination Loss)。
    """
    # 1. 透過 Shared Encoder 獲取 Latent Features
    # 支援 List input (如 SegFormer 接受 pixel_values) 或 unpacked args
    # [NEW] 如果沒有傳入特徵，才需要重新跑 Encoder (節省海量記憶體)
    if latent is None:
        if isinstance(imgs, list) and len(imgs) >= 1:
            encoder_input = imgs[0]
        elif isinstance(imgs, torch.Tensor):
            encoder_input = imgs
        else:
            raise ValueError("imgs must be a list of tensors or a tensor")
        latent = model.forward_encoder(encoder_input)

    # 2. 透過指定任務的 Discriminator 獲取預測
    # model.forward_discriminator 會處理 GRL (Gradient Reversal)
    dis_pred = model.forward_discriminator(latent, task=task)
    
    # 3. Interpolate 到原圖大小以計算 Loss
    dis_pred = F.interpolate(dis_pred, crop_size, mode="bilinear", align_corners=False)

    # 4. 計算 Binary Cross Entropy Loss
    # domain_class_weight[ann] 用於根據語義類別加權 (選擇性)
    # dis_label shape: [B, 1, H, W] or similar
    
    # 確保 dis_label 與 dis_pred 維度匹配
    if dis_pred.shape[1] == 1:
        # BCEWithLogitsLoss
        loss_fct = torch.nn.BCEWithLogitsLoss(reduction='none')
        dis_loss = loss_fct(dis_pred, dis_label.float())
        # ==========================================
        # 套用物件感知遮罩 (Object-Aware Masking)
        # ==========================================
        if spatial_mask is not None:
            if spatial_mask.shape[-2:] != dis_loss.shape[-2:]:
                spatial_mask = F.interpolate(spatial_mask, size=dis_loss.shape[-2:], mode="nearest")
            
            dis_loss = dis_loss * spatial_mask
            mask_sum = spatial_mask.sum()
            
            # 只計算「有物件區域」的平均 Loss
            if mask_sum > 0:
                dis_loss = dis_loss.sum() / mask_sum
            else:
                dis_loss = dis_loss.sum() * 0.0
        else:
            weight = domain_class_weight[ann].unsqueeze(1) if domain_class_weight is not None else None
            if weight is not None:
                dis_loss = (dis_loss * weight).mean()
            else:
                dis_loss = dis_loss.mean()
    else:
        # CrossEntropyLoss (if dis_pred is 2 channels for 0/1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        dis_loss_map = loss_fct(dis_pred, dis_label.squeeze(1).long())
        if domain_class_weight is not None:
             # 注意維度匹配，這裡簡化處理
             weight = domain_class_weight[ann]
             dis_loss = (dis_loss_map * weight).mean()
        else:
            dis_loss = dis_loss_map.mean()

    return dis_loss