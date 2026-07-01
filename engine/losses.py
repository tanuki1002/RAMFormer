"""
engine/losses.py
────────────────────────────────────────────────────────────────────────
通用 Loss 模組。從 train_uda_multitask.py 抽出，避免主訓練腳本膨脹。

Classes:
  MultiTaskLossWrapper  — 自動學習多任務 loss 權重 (Kendall et al.)
  YOLOXLoss             — SimOTA-based YOLOX detection loss (TS & TL 共用)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
# MultiTaskLossWrapper
# ══════════════════════════════════════════════════════════════════════

class MultiTaskLossWrapper(nn.Module):
    """
    透過可學習的 log-variance 自動平衡多任務 loss 權重。
    參考 Kendall et al., "Multi-Task Learning Using Uncertainty."
    """
    def __init__(self, task_num: int = 4):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, losses):
        """一次性計算所有任務的加權 loss（通常逐任務呼叫 get_weighted_loss）。"""
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss = total_loss + precision * loss +  0.5 * self.log_vars[i]
        return total_loss

    def get_weighted_loss(self, loss: torch.Tensor, index: int) -> torch.Tensor:
        """計算單一任務的加權 loss，供逐任務 backward 使用。"""
        precision = torch.exp(-self.log_vars[index])
        return precision * loss + 0.5 * self.log_vars[index]


# ══════════════════════════════════════════════════════════════════════
# YOLOXLoss
# ══════════════════════════════════════════════════════════════════════

class YOLOXLoss(nn.Module):
    """
    YOLOX Detection Loss，整合 SimOTA 正樣本分配。

    支援 Source / Target 兩種模式：
      use_reg_loss=True  (Source)  — IoU loss + L1 loss + Cls + Obj
      use_reg_loss=False (Target)  — 只計算 Cls + Obj（關閉迴歸）
    """

    def __init__(self, num_classes: int, strides: list = None, cls_weights: list = None):
        super().__init__()
        self.num_classes = num_classes
        self.strides     = strides or [8, 16, 32]
        self.bce_loss    = nn.BCEWithLogitsLoss(reduction="none")
        self.l1_loss     = nn.L1Loss(reduction="none")
        self.grids: dict = {}
        # cls_weights: 各類別的分類 loss 權重，用於補償稀有類別（如 TL 的 off）
        # shape [num_classes]，None 表示無加權（全 1.0）
        self.cls_weights = cls_weights

    # ── Forward ────────────────────────────────────────────────────────
    def forward(self, outputs, raw_gt_batch, use_reg_loss: bool = True):
        x_shifts, y_shifts, expanded_strides, origin_preds = [], [], [], []

        for output in outputs:
            stride         = output["stride"]
            cls_pred       = output["cls"]
            reg_pred       = output["reg"]
            obj_pred       = output["obj"]
            batch, _, h, w = cls_pred.shape

            grid_key = f"{h}_{w}"
            if grid_key not in self.grids or self.grids[grid_key].device != cls_pred.device:
                yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
                self.grids[grid_key] = (
                    torch.stack((xv, yv), 2).view(1, 1, h, w, 2).float().to(cls_pred.device)
                )

            grid     = self.grids[grid_key]
            cls_pred = cls_pred.flatten(2).permute(0, 2, 1)
            reg_pred = reg_pred.flatten(2).permute(0, 2, 1)
            obj_pred = obj_pred.flatten(2).permute(0, 2, 1)

            origin_preds.append(torch.cat([reg_pred, obj_pred, cls_pred], dim=2))
            x_shifts.append(grid[..., 0].view(1, -1))
            y_shifts.append(grid[..., 1].view(1, -1))
            expanded_strides.append(
                torch.full((1, grid.shape[2] * grid.shape[3]), stride).to(cls_pred.device)
            )

        outputs_cat          = torch.cat(origin_preds,     dim=1)
        x_shifts_cat         = torch.cat(x_shifts,         dim=1)
        y_shifts_cat         = torch.cat(y_shifts,         dim=1)
        expanded_strides_cat = torch.cat(expanded_strides, dim=1)

        total_loss   = torch.tensor(0.0, device=outputs_cat.device)
        total_num_fg = 0
        batch_size   = outputs_cat.shape[0]

        for b in range(batch_size):
            pred       = outputs_cat[b]
            gt         = raw_gt_batch[b]
            valid_mask = gt[:, 4] != -1
            gt         = gt[valid_mask]

            if gt.shape[0] == 0:
                total_loss = total_loss + self.bce_loss(
                    pred[:, 4], torch.zeros_like(pred[:, 4])
                ).sum()
                continue

            matched_gt_inds, anchor_inds = self.sim_ota(
                pred, gt[:, :4], gt[:, 4],
                x_shifts_cat[0], y_shifts_cat[0], expanded_strides_cat[0],
            )

            num_fg        = len(anchor_inds)
            total_num_fg += num_fg

            obj_target              = torch.zeros(pred.shape[0], device=pred.device)
            obj_target[anchor_inds] = 1.0
            loss_obj                = self.bce_loss(pred[:, 4], obj_target).sum()

            if num_fg > 0:
                if use_reg_loss:
                    matched_pred_box = self.decode_box(
                        pred[anchor_inds, :4],
                        x_shifts_cat[0][anchor_inds],
                        y_shifts_cat[0][anchor_inds],
                        expanded_strides_cat[0][anchor_inds],
                    )
                    iou      = self.compute_iou(matched_pred_box, gt[matched_gt_inds, :4])
                    loss_iou = (1.0 - iou).sum()

                    target_reg = self.get_l1_target(
                        gt[matched_gt_inds, :4],
                        x_shifts_cat[0][anchor_inds],
                        y_shifts_cat[0][anchor_inds],
                        expanded_strides_cat[0][anchor_inds],
                    )
                    loss_l1 = self.l1_loss(pred[anchor_inds, :4], target_reg).sum()
                else:
                    loss_iou = torch.tensor(0.0, device=pred.device)
                    loss_l1  = torch.tensor(0.0, device=pred.device)

                matched_cls_preds = pred[anchor_inds, 5:]
                cls_target        = F.one_hot(gt[matched_gt_inds, 4].long(), self.num_classes).float()
                raw_cls_loss      = self.bce_loss(matched_cls_preds, cls_target)  # [N, C]
                if self.cls_weights is not None:
                    w = torch.tensor(self.cls_weights, dtype=raw_cls_loss.dtype, device=raw_cls_loss.device)
                    raw_cls_loss = raw_cls_loss * w.unsqueeze(0)
                loss_cls = raw_cls_loss.sum()

                if use_reg_loss:
                    total_loss = total_loss + (
                        loss_iou * 5.0 + loss_l1 * 1.0 + loss_obj * 1.0 + loss_cls * 3.0
                    )
                else:
                    # Target domain: 關閉迴歸，只訓練分類與物件存在性
                    total_loss = total_loss + (loss_obj * 1.0 + loss_cls * 3.0)
            else:
                total_loss = total_loss + loss_obj

        # 正規化：除以總正樣本數，解決 loss 爆炸問題
        return total_loss / max(total_num_fg, 1)

    # ── SimOTA ────────────────────────────────────────────────────────
    def sim_ota(self, preds, gt_boxes, gt_classes, x_shifts, y_shifts, strides):
        num_gt        = gt_boxes.shape[0]
        decoded_boxes = self.decode_box(preds[:, :4], x_shifts, y_shifts, strides)

        centers_x = (x_shifts + 0.5) * strides
        centers_y = (y_shifts + 0.5) * strides

        lt       = centers_x.unsqueeze(0) - gt_boxes[:, 0].unsqueeze(1)
        rt       = gt_boxes[:, 2].unsqueeze(1) - centers_x.unsqueeze(0)
        tp       = centers_y.unsqueeze(0) - gt_boxes[:, 1].unsqueeze(1)
        bt       = gt_boxes[:, 3].unsqueeze(1) - centers_y.unsqueeze(0)
        min_dist = torch.stack([lt, rt, tp, bt], dim=-1).min(dim=-1)[0]

        is_in_box  = min_dist > 0.0
        valid_mask = is_in_box.sum(0) > 0
        valid_inds = torch.where(valid_mask)[0]

        if len(valid_inds) == 0:
            return (
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
            )

        valid_preds_box = decoded_boxes[valid_inds]
        valid_preds_cls = preds[valid_inds, 5:]

        iou_cost       = torch.zeros(num_gt, len(valid_inds), device=preds.device)
        pair_wise_ious = torch.zeros(num_gt, len(valid_inds), device=preds.device)
        cls_cost       = torch.zeros(num_gt, len(valid_inds), device=preds.device)

        for i in range(num_gt):
            iou               = self.compute_iou(
                valid_preds_box,
                gt_boxes[i:i+1].repeat(len(valid_inds), 1),
            )
            pair_wise_ious[i] = iou
            iou_cost[i]       = -torch.log(iou + 1e-8)
            gt_cls_logits     = valid_preds_cls[:, gt_classes[i].long()]
            cls_cost[i]       = F.binary_cross_entropy_with_logits(
                gt_cls_logits,
                torch.ones(len(valid_inds), device=preds.device),
                reduction="none",
            )

        cost_matrix = cls_cost + 3.0 * iou_cost + 100000.0 * (~is_in_box[:, valid_inds])
        # 防止 FP16 溢位或權重 NaN 汙染導致 cost_matrix 含 NaN → topk 結果不可信
        cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=-1e6)

        matched_gt_inds, matched_anchor_inds = [], []
        for i in range(num_gt):
            iou_sum = pair_wise_ious[i].sum().item()
            if not math.isfinite(iou_sum):   # NaN / inf → fallback to minimum dynamic_k
                iou_sum = 0.0
            dynamic_k = max(3, min(10, int(iou_sum)))
            dynamic_k = min(dynamic_k, len(valid_inds))
            _, topk_inds = torch.topk(cost_matrix[i], k=dynamic_k, largest=False)
            matched_gt_inds.extend([i] * len(topk_inds))
            matched_anchor_inds.extend(valid_inds[topk_inds].tolist())

        if not matched_gt_inds:
            return (
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
            )

        matched_gt_inds     = torch.tensor(matched_gt_inds,     device=preds.device)
        matched_anchor_inds = torch.tensor(matched_anchor_inds, device=preds.device)

        # 去重：一個 anchor 被多個 GT 選中時，選 cost 最小的
        unique_anchors, counts = torch.unique(matched_anchor_inds, return_counts=True)
        dup_anchors = unique_anchors[counts > 1]
        if len(dup_anchors) > 0:
            keep_mask = torch.ones(len(matched_anchor_inds), dtype=torch.bool, device=preds.device)
            for dup_a in dup_anchors:
                indices   = torch.where(matched_anchor_inds == dup_a)[0]
                valid_idx = (valid_inds == dup_a).nonzero(as_tuple=True)[0][0]
                min_cost, best_idx = 1e9, -1
                for idx in indices:
                    cost = cost_matrix[matched_gt_inds[idx], valid_idx]
                    if cost < min_cost:
                        min_cost = cost
                        best_idx = idx
                for idx in indices:
                    if idx != best_idx:
                        keep_mask[idx] = False
            matched_gt_inds     = matched_gt_inds[keep_mask]
            matched_anchor_inds = matched_anchor_inds[keep_mask]

        return matched_gt_inds, matched_anchor_inds

    # ── Box utilities ─────────────────────────────────────────────────
    def decode_box(self, reg, x_shift, y_shift, stride):
        # No exp for w/h，配合 get_l1_target 的編碼方式
        cx = (reg[:, 0] + x_shift) * stride
        cy = (reg[:, 1] + y_shift) * stride
        w  = (reg[:, 2] * stride).clamp(min=1e-3)
        h  = (reg[:, 3] * stride).clamp(min=1e-3)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

    def get_l1_target(self, gt, x_shift, y_shift, stride):
        # L1 target 是 decode_box 的反函數
        gt_cx = (gt[:, 0] + gt[:, 2]) / 2
        gt_cy = (gt[:, 1] + gt[:, 3]) / 2
        gt_w  = gt[:, 2] - gt[:, 0]
        gt_h  = gt[:, 3] - gt[:, 1]
        tx = gt_cx / stride - x_shift
        ty = gt_cy / stride - y_shift
        tw = gt_w  / stride  # No log
        th = gt_h  / stride  # No log
        return torch.stack([tx, ty, tw, th], dim=1)

    def compute_iou(self, b1, b2):
        inter = (
            (torch.min(b1[:, 2], b2[:, 2]) - torch.max(b1[:, 0], b2[:, 0])).clamp(0)
          * (torch.min(b1[:, 3], b2[:, 3]) - torch.max(b1[:, 1], b2[:, 1])).clamp(0)
        )
        union = (
            (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
          + (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
          - inter
        )
        return inter / (union + 1e-8)

class BinaryIoULoss(nn.Module):
    def __init__(self, ignore_index: int = 255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob  = torch.sigmoid(logits).squeeze(1)
        valid = (targets != self.ignore_index)
        t = targets.float().clone(); t[~valid] = 0.0
        p = prob * valid.float()
        inter = (p * t).sum(dim=(-2, -1))
        union = (p + t - p * t).sum(dim=(-2, -1))
        return (1.0 - (inter + 1.0) / (union + 1.0)).mean()

# 整合雙頭的車道線任務
class HybridLaneLoss(nn.Module):
    """
    Lane Loss = BCE + Dice + IoU + smoothness (+ entropy_min for target).
 
    參數說明：
        pos_weight:  BCE 正例權重（車道線像素稀少，需要加大）
        w_bce:       BCE loss 係數
        w_dice:      Dice loss 係數
        w_iou:       IoU loss 係數（新增，YOLOP 啟發）
        w_smooth:    mask 平滑係數
        w_ent:       target domain entropy minimization 係數
    """
    def __init__(self,
                 pos_weight: float = 10.0,
                 w_bce:      float = 1.0,
                 w_dice:     float = 2.0,
                 w_iou:      float = 1.0,
                 w_smooth:   float = 0.1,
                 w_ent:      float = 0.05):
        super().__init__()
        self.w_bce    = w_bce
        self.w_dice   = w_dice
        self.w_iou    = w_iou
        self.w_smooth = w_smooth
        self.w_ent    = w_ent
        self._pw      = pos_weight
 
        self.iou_loss = BinaryIoULoss(ignore_index=255)
 
    # ── Dice Loss（內部使用）──────────────────────────────────────────
    @staticmethod
    def _dice(logits: torch.Tensor, targets: torch.Tensor,
              ignore_index: int = 255) -> torch.Tensor:
        prob = torch.sigmoid(logits).squeeze(1)          # [B, H, W]
        valid = (targets != ignore_index)
        t = targets.float().clone()
        t[~valid] = 0.0
        p = prob * valid.float()
        inter = (p * t).sum(dim=(-2, -1))
        denom = p.sum(dim=(-2, -1)) + t.sum(dim=(-2, -1))
        return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
 
    # ── Mask 垂直平滑（懲罰鋸齒邊緣）────────────────────────────────
    @staticmethod
    def _smooth(logits: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        dy2  = prob[:, :, 2:, :] - 2 * prob[:, :, 1:-1, :] + prob[:, :, :-2, :]
        return dy2.abs().mean()

    # ── 主 forward ───────────────────────────────────────────────────
    def forward(self,
                outputs:          dict,
                targets_mask:     torch.Tensor,
                is_target_domain: bool = False) -> torch.Tensor:
        """
        Args:
            outputs:          model forward 回傳的 dict，需含 'mask_logits'
            targets_mask:     [B, H, W] float，0=背景，1=車道，255=ignore
            is_target_domain: True → 額外計算 entropy minimization
        """
        logits = outputs['mask_logits']   # [B, 1, H, W]
        sq     = logits.squeeze(1)        # [B, H, W]
 
        # ── BCE ──────────────────────────────────────────────────────
        valid_mask = (targets_mask != 255)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
 
        pw   = torch.tensor([self._pw], dtype=logits.dtype, device=logits.device)
        bce  = F.binary_cross_entropy_with_logits(
            sq[valid_mask],
            targets_mask[valid_mask].float(),
            pos_weight=pw
        )
 
        # ── Dice + IoU ───────────────────────────────────────────────
        dice = self._dice(logits, targets_mask)
        iou  = self.iou_loss(logits, targets_mask)
 
        # ── Smoothness ───────────────────────────────────────────────
        smooth = self._smooth(logits)
 
        total = (self.w_bce    * bce
               + self.w_dice   * dice
               + self.w_iou    * iou
               + self.w_smooth * smooth)
 
        # ── Entropy minimization（target domain 自監督）──────────────
        if is_target_domain:
            prob  = torch.sigmoid(logits)
            eps   = 1e-6
            ent   = -(prob * (prob + eps).log()
                    + (1 - prob) * (1 - prob + eps).log()).mean()
            total = total + self.w_ent * ent
 
        return total

# 基於類別頻率的自適應權重，適用於 RM
class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = logits.float()   # 強制 FP32，防止 AMP FP16 softmax 溢位
        logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
        # logits: [B, C, H, W], targets: [B, H, W]
        valid = (targets != self.ignore_index)
        targets_safe = targets.clone()
        targets_safe[~valid] = 0

        probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
        one_hot = F.one_hot(targets_safe, self.num_classes).permute(0, 3, 1, 2).float()
        one_hot[~valid.unsqueeze(1).expand_as(one_hot)] = 0

        # 只計算前景類（跳過背景 0）
        dice_loss = 0.0
        for c in range(1, self.num_classes):
            p     = probs[:, c] * valid.float()
            g     = one_hot[:, c]
            inter = (p * g).sum()
            union = p.sum() + g.sum()
            dice_loss += 1.0 - (2 * inter + self.smooth) / (union + self.smooth)
        return dice_loss / max(self.num_classes - 1, 1)


# ══════════════════════════════════════════════════════════════════════
# PrototypeContrastiveLoss
# ══════════════════════════════════════════════════════════════════════
class PrototypeContrastiveLoss(nn.Module):
    """
    Prototype-based contrastive loss for semantic segmentation.

    為每個語意類別維護一個 prototype（EMA 平均特徵方向），
    訓練時對採樣到的 pixel feature 計算 InfoNCE loss：
      L = -log( exp(cos(f, proto[y]) / τ) / Σ_c exp(cos(f, proto[c]) / τ) )

    直接拉大相似類別（如各種箭頭）在 feature space 的距離，
    解決「找到區域但分類錯誤」的問題。

    Args:
        num_classes:   語意類別數（含背景）
        feat_dim:      feature 維度，需與 pre_cls 輸出一致（預設 128）
        temperature:   InfoNCE 溫度 τ，越小對錯誤類別懲罰越重
        momentum:      prototype EMA 動量（0.999 = 每 1000 batch 約更新一次）
        ignore_index:  GT label 中的 ignore 值
        max_samples:   每個類別每 batch 最多採樣的 pixel 數，避免記憶體爆炸
    """
    def __init__(self,
                 num_classes:  int,
                 feat_dim:     int   = 128,
                 temperature:  float = 0.07,
                 momentum:     float = 0.999,
                 ignore_index: int   = 255,
                 max_samples:  int   = 128):
        super().__init__()
        self.num_classes  = num_classes
        self.temperature  = temperature
        self.momentum     = momentum
        self.ignore_index = ignore_index
        self.max_samples  = max_samples
        # prototype bank：不是可訓練參數，但跟著模型 .to(device)
        self.register_buffer('prototypes',  torch.zeros(num_classes, feat_dim))
        self.register_buffer('initialized', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def _update(self, feats_norm: torch.Tensor, labels: torch.Tensor):
        """EMA 更新 prototype bank（no grad）。"""
        for c in range(self.num_classes):
            mask = (labels == c)
            if not mask.any():
                continue
            mean_feat = F.normalize(feats_norm[mask].mean(0), dim=0)
            if not self.initialized[c]:
                self.prototypes[c] = mean_feat
                self.initialized[c] = True
            else:
                self.prototypes[c] = (self.momentum * self.prototypes[c]
                                      + (1.0 - self.momentum) * mean_feat)
                self.prototypes[c] = F.normalize(self.prototypes[c], dim=0)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats:  [B, C, H, W]  pre_cls feature（1/4 scale）
            labels: [B, H, W]     GT label（1/4 scale，long tensor）
        Returns:
            scalar contrastive loss
        """
        B, C, H, W = feats.shape
        feats_flat  = feats.permute(0, 2, 3, 1).reshape(-1, C).float()   # [N, C]
        labels_flat = labels.reshape(-1)                                   # [N]

        valid    = labels_flat != self.ignore_index
        feats_v  = feats_flat[valid]
        labels_v = labels_flat[valid]

        if feats_v.shape[0] == 0:
            return feats_flat.sum() * 0.0   # feats_flat 已是 float32

        feats_norm = F.normalize(feats_v, dim=1)                          # [Nv, C]

        # prototype EMA update（no grad）
        self._update(feats_norm.detach(), labels_v)

        # 至少需要 2 個 prototype 才能計算對比 loss
        if self.initialized.sum() < 2:
            return feats_flat.sum() * 0.0

        # 每個類別隨機採樣 max_samples 個 pixel
        idx_list = []
        for c in range(self.num_classes):
            if not self.initialized[c]:
                continue
            idx = (labels_v == c).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            if idx.numel() > self.max_samples:
                perm = torch.randperm(idx.numel(), device=idx.device)[:self.max_samples]
                idx  = idx[perm]
            idx_list.append(idx)

        if not idx_list:
            return feats_flat.sum() * 0.0

        idx_all  = torch.cat(idx_list)
        s_feats  = feats_norm[idx_all]                                    # [S, C]
        s_labels = labels_v[idx_all]                                      # [S]

        protos = F.normalize(self.prototypes, dim=1)                      # [K, C]
        sim    = torch.mm(s_feats, protos.T) / self.temperature           # [S, K]

        # 未初始化的 prototype 不參與 softmax 分母
        sim[:, ~self.initialized] = -1e4

        return F.cross_entropy(sim, s_labels)
