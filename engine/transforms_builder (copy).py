"""
engine/transforms_builder.py
────────────────────────────────────────────────────────────────────────
集中管理各任務的 Transform pipeline 建構邏輯。
原本散落在 train_uda_multitask.py main() 內的 get_transforms /
get_ll_transforms 移到這裡，讓主訓練腳本只做「呼叫」。

原邏輯完全不變，僅移位。

Functions:
  build_rm_transforms  — Road Marking (RLMD) 任務用
  build_ll_transforms  — Lane Line (BDD100K) 任務用

Note:
  TS / TL 的前處理由 dataloader 內部完成（transforms=None），
  因此此檔不提供對應 builder。

Helper classes（原 train_uda_multitask.py 頂層）:
  LoadBDDColorLabelToID  — BDD100K 彩色 label 轉 class-ID
  Cleanup                — 移除 data dict 中值為 None 的 key
"""

from PIL import Image
import numpy as np
import torch
import math
from engine import transform
import random

class NightRainAug:
    """
    模擬夜晚+雨天 appearance，縮小 source (晴天) → target (夜晚/雨天) 的 domain gap。
    在 ToTensor() 之後、Normalize() 之前使用，img 為 [0,1] float tensor。
    """
    def __init__(self, p_night=0.35, p_rain=0.35):
        self.p_night = p_night
        self.p_rain  = p_rain

    def _aug_tensor(self, img, do_night, gamma, rain_params):
        if do_night:
            noise = torch.randn_like(img) * 0.03
            img = (img.clamp(0.0, 1.0) ** gamma + noise).clamp(0.0, 1.0)
        for x1, x2, intensity in rain_params:
            h = img.shape[-2]
            img[:, h // 2:, x1:x2] = (img[:, h // 2:, x1:x2] + intensity).clamp(0.0, 1.0)
        return img

    def transform(self, data):
        # Determine aug params once so all views are augmented consistently
        do_night = random.random() < self.p_night
        gamma    = random.uniform(1.4, 2.8) if do_night else 1.0

        do_rain     = random.random() < self.p_rain
        rain_params = []
        if do_rain:
            ref = data["imgs"][0] if "imgs" in data else data.get("img")
            if ref is None:
                return data
            w = ref.shape[-1]
            for _ in range(random.randint(2, 6)):
                x     = random.randint(0, w - 1)
                width = random.randint(1, 3)
                rain_params.append((max(0, x - width), min(w, x + width),
                                    random.uniform(0.15, 0.45)))

        if not do_night and not do_rain:
            return data

        if "imgs" in data:
            data["imgs"] = [self._aug_tensor(img, do_night, gamma, rain_params)
                            for img in data["imgs"]]
        else:
            img = data.get("img")
            if img is not None:
                data["img"] = self._aug_tensor(img, do_night, gamma, rain_params)
        return data

class LoadBDDColorLabelToID:
    def transform(self, data):
        # 處理部分缺失標籤的情況：如果沒有標籤路徑，生成全 255 的 Dummy Mask
        if data.get("ann_path") is None:
            # 嘗試從已讀取的圖片獲取尺寸
            if "img" in data:
                if isinstance(data["img"], torch.Tensor):
                    h, w = data["img"].shape[-2:]
                elif isinstance(data["img"], np.ndarray): # cv2 img
                    h, w = data["img"].shape[:2]
                else:
                    h, w = 512, 512 # Fallback
            else:
                h, w = 512, 512 # Fallback   
            # 建立全為 255 (Ignore Index) 的 Mask
            data["ann"] = torch.full((1, h, w), 255, dtype=torch.long)
            return data
        
        # 讀取圖片 (P 或 L 模式)
        ann = Image.open(data["ann_path"])
        ann_np = np.asarray(ann).copy()
        # 準備輸出的 ID Map (預設全為 0/背景)
        # 這裡產生的就是模型要吃的 LongTensor 格式 (Class IDs)
        id_map = np.zeros(ann_np.shape[:2], dtype=np.int64)

        if len(ann_np.shape) == 3:
            # RGB 邏輯: 任何有顏色的地方設為 1
            mask_foreground = (ann_np[:, :, 0] > 100) | \
                              (ann_np[:, :, 1] > 100) | \
                              (ann_np[:, :, 2] > 100)
            id_map[mask_foreground] = 1
        else:
            # === 這裡是重點：處理 P/L 模式的單通道數值 ===            
            # 情況 B: 車道線 (LL) CSV 定義 Lane = 255
            # 注意：原本 255 是 Ignore Index，但這裡它是"線"，所以必須轉成 ID 1
            mask_255 = (ann_np == 255)
            id_map[mask_255] = 1
            
            # 備註：背景 0 已經在初始化時設好了，不用動

        # 轉回 PyTorch Tensor [1, H, W]，這就是模型要吃的格式
        data["ann"] = torch.from_numpy(id_map)[None, :].long()
        return data

class Cleanup:
    def transform(self, data):
        # 複製 key 列表以避免在迭代時刪除
        keys = list(data.keys())
        for k in keys:
            if data[k] is None:
                del data[k]
        return data

class FourierDomainAug:
    """
    Fourier Domain Adaptation。
    target_dirs: 目標域圖片的「資料夾路徑」list（自動掃描 jpg/png）。
    cache_size:  預載圖片數量，避免每次 I/O。
    p:           觸發機率。
    """
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, target_dirs: list, beta: float = 0.01,
                 p: float = 0.5, cache_size: int = 100, max_weight: float = 0.75):
        import os
        self.beta       = beta
        self.p          = p
        self.max_weight = max_weight

        # 掃描目錄，收集所有圖片路徑
        all_paths = []
        for d in target_dirs:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if os.path.splitext(f)[1].lower() in self._IMG_EXTS:
                        all_paths.append(os.path.join(d, f))
        if not all_paths:
            self._cache = []
            return

        # 預載隨機抽樣的 cache_size 張圖（避免每次 disk I/O）
        sampled = random.sample(all_paths, min(cache_size, len(all_paths)))
        self._cache = []
        for p_ in sampled:
            try:
                img = np.array(Image.open(p_).convert("RGB"), dtype=np.float32) / 255.0
                self._cache.append(torch.from_numpy(img).permute(2, 0, 1))  # [C,H,W]
            except Exception:
                pass

    def transform(self, data):
        if not self._cache or random.random() >= self.p:
            return data
        src = data.get("img")
        if src is None:
            return data

        tgt = random.choice(self._cache)
        if tgt.shape[-2:] != src.shape[-2:]:
            tgt = torch.nn.functional.interpolate(
                tgt.unsqueeze(0), size=src.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(0)

        h, w = src.shape[-2:]
        b = int(min(h, w) * self.beta)
        if b == 0:
            return data

        src_shift = torch.fft.fftshift(torch.fft.fft2(src))
        tgt_shift = torch.fft.fftshift(torch.fft.fft2(tgt))

        src_amp = src_shift.abs()
        src_pha = src_shift.angle()
        cy, cx  = h // 2, w // 2

        # 圓形遮罩：以歐氏距離界定低頻區域，餘弦漸進權重（等向性）
        ys = torch.arange(h, dtype=src.dtype, device=src.device) - cy
        xs = torch.arange(w, dtype=src.dtype, device=src.device) - cx
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt(grid_y ** 2 + grid_x ** 2)

        mask   = r <= b
        weight = torch.zeros(h, w, dtype=src.dtype, device=src.device)
        weight[mask] = (1.0 + torch.cos(math.pi * r[mask] / b)) / 2.0 * self.max_weight
        weight = weight.unsqueeze(0)  # [1, H, W]

        tgt_amp = tgt_shift.abs()
        src_amp = weight * tgt_amp + (1.0 - weight) * src_amp

        data["img"] = torch.fft.ifft2(
            torch.fft.ifftshift(src_amp * torch.exp(1j * src_pha))
        ).real.clamp(0, 1)
        return data

# ══════════════════════════════════════════════════════════════════════
# Public builders
# ══════════════════════════════════════════════════════════════════════
def build_rm_transforms(cfg, is_train: bool, is_target: bool = False,
                        is_val: bool = False, use_bdd_color_fix: bool = False):
    """
    建立 Road Marking (RLMD) 任務的 transform pipeline。

    Args:
        cfg:              MultiTaskTrainingConfig
        is_train:         訓練模式
        is_target:        是否為 target domain（無 GT 標籤）
        is_val:           是否為驗證集
        use_bdd_color_fix: 是否使用 BDD100K 彩色轉 ID 的 Loader
    """
    t_list = [transform.LoadImg()]

    if not is_target or is_val:
        if not (is_target and not is_val):
            if use_bdd_color_fix:
                t_list.append(LoadBDDColorLabelToID())
            else:
                t_list.append(transform.LoadAnn())

    if cfg.contrast_stretch:
        for idx in range(len(cfg.contrast_stretch)):
            t_list.append(transform.ContrastStretch(
                max_intensity=cfg.max_intensity,
                function_name=cfg.contrast_stretch[idx],
                parameter=cfg.img_proc_params[idx],
            ))

    t_list.append(transform.ToTensor())

    if is_train:
        t_list.append(transform.RandomResizeCrop(cfg.image_scale, cfg.random_resize_ratio, cfg.crop_size))
        if not is_target:  # Source：顏色與模糊增強
            t_list.append(transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))
            t_list.append(transform.RandomGaussian(kernel_size=5))
        else:  # Target：外觀增強 + 遮擋一致性增強
            t_list.append(NightRainAug(p_night=0.3, p_rain=0.2))
            for _ in range(cfg.num_masks):
                t_list.append(transform.RandomErase(scale=(0.02, 0.04)))
    else:
        t_list.append(transform.Resize(cfg.image_scale))

    t_list.append(transform.Normalize())
    t_list.append(transform.Check())
    t_list.append(Cleanup())
    return t_list


def build_ll_transforms(cfg, is_train: bool, is_target: bool = False):
    t_list = [transform.LoadImg()]
    t_list.append(transform.ToTensor())
    t_list.append(transform.Resize(cfg.crop_size))

    if is_train and not is_target:
        t_list.append(transform.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))
        t_list.append(transform.RandomGaussian(kernel_size=5))
        # NightRainAug 雨滴條紋（局部效果）、全局暗化
        t_list.append(NightRainAug(p_night=0.2, p_rain=0.15))

    t_list.append(transform.Normalize())
    t_list.append(Cleanup())
    return t_list
