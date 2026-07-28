"""
train_uda_multitask.py
────────────────────────────────────────────────────────────────────────
主訓練入口。原本 1700+ 行的腳本，透過以下拆分精簡：
 
（放在 engine/ 底下）：
    engine/losses.py              ← YOLOXLoss、MultiTaskLossWrapper
    engine/decode_utils.py        ← decode_yolox_outputs、format_pseudo_labels_to_gt
    engine/visualization.py       ← visualize_*
    engine/transforms_builder.py  ← build_rm_transforms、build_ll_transforms
                                     LoadBDDColorLabelToID、Cleanup
"""
import sys
import os
import gc
import time
import random
import cv2
from datetime import datetime, timedelta
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch_optimizer import Lookahead
from ema_pytorch import EMA
from transformers import SegformerConfig
 
from engine.multi_task_segformer import get_model as get_mt_model
from engine.dataloader import get_dataset, RareCategoryManager, InfiniteDataloader
from engine.category import Category
from engine.multi_uda import (
    PixelThreshold, MultiTaskClassMix,
    compute_domain_discrimination_loss_mt,
)
from engine import transform
from engine.misc import set_seed
from engine.metric import Metrics
from engine.validator import Validator
from engine.det_metric import DetectionMetric
 
from engine.losses import YOLOXLoss, MultiTaskLossWrapper, HybridLaneLoss, DiceLoss, PrototypeContrastiveLoss
from engine.decode_utils import (
    decode_yolox_outputs,
)
from engine.visualization import (
    visualize_validation_samples,
    visualize_ts_predictions,
    visualize_lstr_predictions,
)
from engine.transforms_builder import (
    build_rm_transforms,
    build_ll_transforms,
    FourierDomainAug,
)

from configs.multitask_config import MultiTaskTrainingConfig

def get_boundary_map(ann: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    """
    從語意標注產生二元邊界圖：任何與相鄰像素（上下左右）類別不同的位置標記為 1。
    ignore_index 的像素不產生邊界，且不被視為有效鄰居。
    Args:
        ann: [B, H, W] long tensor，類別標注
    Returns:
        boundary: [B, H, W] float32 tensor，1 = 邊界，0 = 非邊界
    """
    ig = ignore_index
    boundary = torch.zeros_like(ann, dtype=torch.float32)
    valid = (ann != ig)
    for shift_h, shift_w in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = torch.full_like(ann, ig)
        if shift_h == -1:
            shifted[:, :-1, :] = ann[:, 1:, :]
        elif shift_h == 1:
            shifted[:, 1:, :] = ann[:, :-1, :]
        elif shift_w == -1:
            shifted[:, :, :-1] = ann[:, :, 1:]
        elif shift_w == 1:
            shifted[:, :, 1:] = ann[:, :, :-1]
        boundary += ((ann != shifted) & valid & (shifted != ig)).float()
    return (boundary > 0).float()


class Logger(object):
    """將 stdout 同時輸出到 Terminal 和 log 檔。"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
 
    def flush(self):
        self.terminal.flush()
        self.log.flush()

class TaskModelWrapper(nn.Module):
    """
    包裝 MultiTaskSegformer，使其符合 Validator 的呼叫介面。
    Validator 預期回傳 (logits, loss) tuple，但模型回傳 Dict。
    """
    def __init__(self, model, task):
        super().__init__()
        self.model = model
        self.task  = task
 
    def forward(self, images=None, pixel_values=None, **kwargs):
        img_input = pixel_values if pixel_values is not None else images
        if isinstance(img_input, list):
            img_input = img_input[0]
        outputs = self.model(pixel_values=img_input, task=self.task, **kwargs)
        if isinstance(outputs, dict):
            logits = outputs.get("mask_logits", outputs.get("logits"))
            return logits, outputs.get("loss", None)
        return outputs
    
strong_pixel_aug = T.Compose([
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
])

def validate_det_task(model, dataloader, device, num_classes, task="ts", nms_thresh=0.50):
    """計算偵測任務 (TS/TL) 的 mAP。"""
    print(f"Validating Detection ({task.upper()}) (mAP)...")
    metric = DetectionMetric(num_classes=num_classes)
    model.eval()
 
    is_infinite = hasattr(dataloader, "iterator")
    max_batches = 100 if is_infinite else len(dataloader)
 
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= max_batches:
                break
            imgs = data["img"].to(device)
            if "raw_gt" not in data:
                continue
            gt_data = data["raw_gt"].to(device)
 
            outputs = model(pixel_values=imgs, task=task)["logits"]
            final_bboxes, final_scores, final_classes = decode_yolox_outputs(
                outputs, conf_thresh=0.05, nms_thresh=nms_thresh
            )
 
            for b in range(imgs.shape[0]):
                if len(final_bboxes[b]) == 0:
                    continue
                p_boxes  = final_bboxes[b].cpu().numpy()
                p_scores = final_scores[b].cpu().numpy()
                p_clses  = final_classes[b].cpu().numpy()
 
                g_boxes = gt_data[b][:, :4].cpu().numpy()
                g_clses = gt_data[b][:,  4].cpu().numpy()
                valid   = g_clses != -1
                metric.update(p_boxes, p_clses, p_scores, g_boxes[valid], g_clses[valid])
 
    return metric.compute_map()
 
 
def validate_ll_task(model, dataloader, criterion, device):
    """計算 Lane Segmentation 在驗證集上的平均 Binary IoU。"""
    print("Validating Lane Line (LL) (Binary IoU)...")
    model.eval()
    total_iou, valid_batches = 0.0, 0
 
    with torch.no_grad():
        for data in dataloader:
            imgs = data["img"].to(device)
            if "lane_mask" not in data:
                continue
            gt_mask = data["lane_mask"].to(device)
 
            outputs_dict = model(pixel_values=imgs, task="ll")
            outputs = outputs_dict.get("mask_logits", outputs_dict.get("logits"))
 
            if gt_mask.shape[-2:] != outputs.shape[-2:]:
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(1).float(),
                    size=outputs.shape[-2:],
                    mode="nearest",
                ).squeeze(1)
 
            pred_mask     = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            intersection  = (pred_mask * gt_mask).sum(dim=(1, 2))
            union         = ((pred_mask + gt_mask) > 0).float().sum(dim=(1, 2))
            iou           = (intersection / (union + 1e-6)).mean().item()
            total_iou    += iou
            valid_batches += 1
 
    return total_iou / max(1, valid_batches)

def main(cfg: MultiTaskTrainingConfig, exp_name: str, checkpoint: str, log_dir: str):
 
    currentTime = datetime.now().strftime("%Y%m%d%H%M%S") if log_dir is None else log_dir[-14:]
    tb_dir      = f"logs/{exp_name}_{currentTime}" if log_dir is None else f"logs/{log_dir}"
    os.makedirs(tb_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(tb_dir, "log.txt"))
    print(f"Starting training session: {exp_name}")
    print(f"Logs will be saved to: {os.path.join(tb_dir, 'log.txt')}")
 
    writer      = SummaryWriter(tb_dir)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device      = torch.device(device_name)
    set_seed(cfg.seed)
 
    cats_rm = Category.load(cfg.category_csv_rlmd)
    cats_ll = Category.load(cfg.category_csv_ll)
    
    if cfg.category_csv_ts and os.path.exists(cfg.category_csv_ts):
        cats_ts = Category.load(cfg.category_csv_ts)
        print(f"Loaded {len(cats_ts)} TS categories from {cfg.category_csv_ts}")
    else:
        cats_ts = []
        print("Warning: TS category CSV not found.")
    if len(cats_ts) > 0:
        cfg.ts_num_classes = len(cats_ts)
 
    palette_rm = np.array([[c.r, c.g, c.b] for c in cats_rm], dtype=np.uint8)
    palette_ll = np.array([[c.r, c.g, c.b] for c in cats_ll], dtype=np.uint8)
 
    rcm_rm = RareCategoryManager(cats_rm, cfg.rcs_path_rlmd, cfg.rcs_temperature) if cfg.rcs_path_rlmd else None
    rcm_ll = RareCategoryManager(cats_ll, cfg.rcs_path_ll,   cfg.rcs_temperature) if cfg.rcs_path_ll   else None

    ll_class_weights = torch.ones(len(cats_ll)).to(device)
    ll_class_weights[1] = 2.0
 
    transforms_rm_train_src = build_rm_transforms(cfg, is_train=True,  is_target=False)
    transforms_rm_train_tgt = build_rm_transforms(cfg, is_train=True,  is_target=True)
    transforms_rm_val       = build_rm_transforms(cfg, is_train=False, is_val=True)
 
    transforms_ll_train_src = build_ll_transforms(cfg, is_train=True,  is_target=False)
    transforms_ll_train_tgt = build_ll_transforms(cfg, is_train=True,  is_target=True)
    transforms_ll_val       = build_ll_transforms(cfg, is_train=False)
 
    transforms_ts = None
 
    ds_rm_src_train = get_dataset(cfg.dataset_rlmd, cfg.source_train_images_rlmd, cfg.source_train_labels_rlmd, rcm_rm, transforms_rm_train_src)
    ds_rm_tgt_train = get_dataset(cfg.dataset_rlmd, cfg.target_train_images_rlmd, None,                         None,   transforms_rm_train_tgt)
    ds_rm_src_val   = get_dataset(cfg.dataset_rlmd, cfg.source_val_images_rlmd,   cfg.source_val_labels_rlmd,   None,   transforms_rm_val)
    ds_rm_tgt_val   = get_dataset(cfg.dataset_rlmd, cfg.target_val_images_rlmd,   cfg.target_val_labels_rlmd,   None,   transforms_rm_val)
 
    ds_ll_src_train = get_dataset(cfg.dataset_ll, cfg.source_train_images_ll, cfg.source_train_labels_ll, rcm_ll, transforms_ll_train_src)
    ds_ll_tgt_train = get_dataset(cfg.dataset_ll, cfg.target_train_images_ll, None,                       None,   transforms_ll_train_tgt)
    ds_ll_src_val   = get_dataset(cfg.dataset_ll, cfg.source_val_images_ll,   cfg.source_val_labels_ll,   None,   transforms_ll_val)
    ds_ll_tgt_val   = get_dataset(cfg.dataset_ll, cfg.target_val_images_ll,   cfg.target_val_labels_ll,   None,   transforms_ll_val)
    
    TS_INPUT_SIZE      = (960, 960)
    CROPPED_SIGNS_DIR  = "/home/rvl/MinHsuan/dataset/Traffic Sign/TT100k/tt100k_only_500/ClassMix_cropped"
    CROPPED_LIGHTS_DIR = "/home/rvl/MinHsuan/dataset/Traffic Light/S2TLD_依照天氣整理/clear/ClassMix_cropped"
 
    ds_ts_src_train = get_dataset(
        cfg.dataset_ts, cfg.source_train_images_ts, cfg.source_train_labels_ts,
        rcm=None, transforms=transforms_ts, input_size=TS_INPUT_SIZE,
        is_train=True, cropped_data_dir=CROPPED_SIGNS_DIR,
    )
    ds_ts_tgt_train = (
        get_dataset(cfg.dataset_ts, cfg.target_train_images_ts, None,
                    rcm=None, transforms=transforms_ts, input_size=TS_INPUT_SIZE, is_train=False)
        if cfg.target_train_images_ts else None
    )
    ds_ts_src_val = (
        get_dataset(cfg.dataset_ts, cfg.source_val_images_ts, cfg.source_val_labels_ts,
                    rcm=None, transforms=transforms_ts, input_size=TS_INPUT_SIZE)
        if cfg.source_val_images_ts else None
    )
    ds_ts_tgt_val = (
        get_dataset(cfg.dataset_ts, cfg.target_val_images_ts, cfg.target_val_labels_ts,
                    rcm=None, transforms=transforms_ts, input_size=TS_INPUT_SIZE)
        if cfg.target_val_images_ts else None
    )
 
    ds_tl_src_train = get_dataset(
        cfg.dataset_tl, cfg.source_train_images_tl, cfg.source_train_labels_tl,
        rcm=None, transforms=transforms_ts, input_size=TS_INPUT_SIZE,
        is_train=True, cropped_data_dir=CROPPED_LIGHTS_DIR,
    )
    ds_tl_tgt_train = (
        get_dataset(cfg.dataset_tl, cfg.target_train_images_tl, None,
                    rcm=None, transforms=transforms_ts, input_size=TS_INPUT_SIZE, is_train=False)
        if cfg.target_train_images_tl else None
    )
    ds_tl_src_val = (
        get_dataset(cfg.dataset_tl, cfg.source_val_images_tl, cfg.source_val_labels_tl,
                    rcm=None, transforms=transforms_ts, input_size=TS_INPUT_SIZE)
        if cfg.source_val_images_tl else None
    )
    ds_tl_tgt_val = (
        get_dataset(cfg.dataset_tl, cfg.target_val_images_tl, cfg.target_val_labels_tl,
                    rcm=None, transforms=transforms_ts, input_size=TS_INPUT_SIZE)
        if cfg.target_val_images_tl else None
    )

 
    dl_rm_src = InfiniteDataloader(ds_rm_src_train, cfg.train_batch_size, True, cfg.num_workers, True, cfg.pin_memory)
    dl_rm_tgt = InfiniteDataloader(ds_rm_tgt_train, cfg.train_batch_size, True, cfg.num_workers, True, cfg.pin_memory)
    dl_ll_src = InfiniteDataloader(ds_ll_src_train, cfg.train_batch_size, True, cfg.num_workers, True, cfg.pin_memory)
    dl_ll_tgt = InfiniteDataloader(ds_ll_tgt_train, cfg.train_batch_size, True, cfg.num_workers, True, cfg.pin_memory)
    dl_ts_src = InfiniteDataloader(ds_ts_src_train, cfg.train_batch_size, True, cfg.num_workers, True, cfg.pin_memory)
    dl_tl_src = InfiniteDataloader(ds_tl_src_train, cfg.train_batch_size, True, cfg.num_workers, True, cfg.pin_memory)
    dl_ts_tgt = InfiniteDataloader(ds_ts_tgt_train, cfg.train_batch_size, True, cfg.num_workers, True, cfg.pin_memory) if ds_ts_tgt_train else None
    dl_tl_tgt = InfiniteDataloader(ds_tl_tgt_train, cfg.train_batch_size, True, cfg.num_workers, True, cfg.pin_memory) if ds_tl_tgt_train else None
 
    val_loader_kwargs = {"batch_size": cfg.val_batch_size, "shuffle": False, "num_workers": 1, "pin_memory": cfg.pin_memory}
    dl_rm_src_val = DataLoader(ds_rm_src_val, **val_loader_kwargs)
    dl_rm_tgt_val = DataLoader(ds_rm_tgt_val, **val_loader_kwargs)
    dl_ll_src_val = DataLoader(ds_ll_src_val, **val_loader_kwargs)
    dl_ll_tgt_val = DataLoader(ds_ll_tgt_val, **val_loader_kwargs)
    dl_ts_src_val = DataLoader(ds_ts_src_val, **val_loader_kwargs) if ds_ts_src_val else None
    dl_ts_tgt_val = DataLoader(ds_ts_tgt_val, **val_loader_kwargs) if ds_ts_tgt_val else None
    dl_tl_src_val = DataLoader(ds_tl_src_val, **val_loader_kwargs) if ds_tl_src_val else None
    dl_tl_tgt_val = DataLoader(ds_tl_tgt_val, **val_loader_kwargs) if ds_tl_tgt_val else None

 
    config_rm = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_rm.num_labels = len(cats_rm)
    config_rm.semantic_loss_ignore_index = cfg.ignore_index[0][0] if cfg.ignore_index[0] else 255

    config_ll = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_ll.num_labels = len(cats_ll)
    config_ll.semantic_loss_ignore_index = cfg.ignore_index[1][0] if cfg.ignore_index[1] else 255

    config_ts = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_ts.num_labels = cfg.ts_num_classes

    config_tl = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_tl.num_labels = cfg.tl_num_classes
 
    model = get_mt_model(config_rm, config_ll, config_ts, config_tl)
    ema = EMA(model, beta=0.999, update_after_step=1500)
 
    initial_log_vars = torch.tensor([0.0, 0.5, 1.5, 2.0])
    
    mt_loss_wrapper  = MultiTaskLossWrapper(task_num=4).to(device)
    with torch.no_grad():
        mt_loss_wrapper.log_vars.data = initial_log_vars
 
    def _split_no_wd(module):
        with_wd, no_wd = [], []
        for name, p in module.named_parameters():
            if p.ndim <= 1 or name.endswith("bias") or "alpha" in name:
                no_wd.append(p)
            else:
                with_wd.append(p)
        return with_wd, no_wd

    ts_wd, ts_no_wd = _split_no_wd(model.decoder_ts)
    tl_wd, tl_no_wd = _split_no_wd(model.decoder_tl)

    base_optimizer = optim.AdamW([
        {"name": "backbone",     "params": model.encoder.parameters(),          "lr": cfg.backbone_lr, "weight_decay": cfg.weight_decay},
        {"name": "head_rm",      "params": model.decoder_rm.parameters(),       "lr": cfg.head_lr_rm if cfg.head_lr_rm is not None else cfg.head_lr, "weight_decay": cfg.weight_decay},
        {"name": "head_ll",      "params": model.decoder_ll.parameters(),       "lr": cfg.head_lr,     "weight_decay": cfg.weight_decay},
        {"name": "head_ts",      "params": ts_wd,                               "lr": cfg.head_lr,     "weight_decay": cfg.weight_decay},
        {"name": "head_ts_nowd", "params": ts_no_wd,                            "lr": cfg.head_lr,     "weight_decay": 0.0},
        {"name": "head_tl",      "params": tl_wd,                               "lr": cfg.head_lr,     "weight_decay": cfg.weight_decay},
        {"name": "head_tl_nowd", "params": tl_no_wd,                            "lr": cfg.head_lr,     "weight_decay": 0.0},
        {"name": "disc_rm",      "params": model.discriminator_rm.parameters(), "lr": cfg.head_lr,     "weight_decay": cfg.weight_decay},
        {"name": "disc_ll",      "params": model.discriminator_ll.parameters(), "lr": cfg.head_lr,     "weight_decay": cfg.weight_decay},
        {"name": "disc_ts",      "params": model.discriminator_ts.parameters(), "lr": cfg.head_lr,     "weight_decay": cfg.weight_decay},
        {"name": "disc_tl",      "params": model.discriminator_tl.parameters(), "lr": cfg.head_lr,     "weight_decay": cfg.weight_decay},
        {"name": "loss_params",  "params": mt_loss_wrapper.parameters(),        "lr": 1e-4,            "weight_decay": 0.0},
    ])
    optimizer = Lookahead(optimizer=base_optimizer, k=5, alpha=0.5)
 
    metric_rm = Metrics(num_categories=len(cats_rm), nan_to_num=0)
    if hasattr(metric_rm, "ignore_index"):
        metric_rm.ignore_index = 255
    elif hasattr(metric_rm, "ignore_ids"):
        metric_rm.ignore_ids = [255]
 
    model_wrapper_rm = TaskModelWrapper(model, task="rm")
 
    class_ignore_rm  = cfg.ignore_index[0] if hasattr(cfg, "ignore_index") and len(cfg.ignore_index) > 0 else []
    validator_rm_src = Validator(dl_rm_src_val, model_wrapper_rm, device, metric_rm, cfg.crop_size, cfg.stride, len(cats_rm), "slide", class_ignore_rm)
    validator_rm_tgt = Validator(dl_rm_tgt_val, model_wrapper_rm, device, metric_rm, cfg.crop_size, cfg.stride, len(cats_rm), "slide", class_ignore_rm)
 
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer.optimizer, 1e-4, 1, 1500)
    poly_scheduler   = torch.optim.lr_scheduler.PolynomialLR(optimizer.optimizer, cfg.max_iters - 1500, 1)
    scheduler        = torch.optim.lr_scheduler.SequentialLR(
        optimizer.optimizer,
        schedulers=[warmup_scheduler, poly_scheduler],
        milestones=[1500],
    )
 
    rm_ce_weight = torch.ones(len(cats_rm), device=device)
    rm_ce_weight[0]  = 0.1
    rm_ce_weight[1]  = 1.5
    rm_ce_weight[11] = 1.3
    rm_ce_weight[12] = 1.3
    rm_ce_weight[13] = 1.3
    rm_ce_weight[14] = 1.3
    rm_ce_weight[15] = 1.3
    dice_loss_rm_src  = DiceLoss(num_classes=len(cats_rm), ignore_index=255)
    proto_loss_rm     = PrototypeContrastiveLoss(num_classes=len(cats_rm)).to(device)

    rm_class_thresholds = [0.95] + [0.90] * (len(cats_rm) - 1)

    target_criterion = PixelThreshold(threshold=0.968, focal_loss=None)
    dl_classmix_src = InfiniteDataloader(ds_rm_src_train, 1, True, cfg.num_workers, True, cfg.pin_memory)
    classmix_rm     = MultiTaskClassMix(device, cfg.train_batch_size, rcm_rm, cats_rm, dl_classmix_src, dl_rm_tgt, ema, cfg.mix_num)
    scaler           = torch.amp.GradScaler("cuda", enabled=False)

    criterion_det_ts = YOLOXLoss(num_classes=cfg.ts_num_classes).to(device)
    criterion_det_tl = YOLOXLoss(num_classes=cfg.tl_num_classes).to(device)
    criterion_hybrid = HybridLaneLoss().to(device)

    fda_ts = FourierDomainAug(
    target_dirs=[cfg.target_train_images_ts], beta=0.005, p=1.0, cache_size=100,)

    model.to(device)
    print("Applying Fine-tuning Strategy: Freezing unused branches...")
    task_weights = cfg.task_weight if isinstance(cfg.task_weight, dict) else vars(cfg.task_weight)
 
    if task_weights.get("rm", 1.0) == 0.0:
        print("  -> 凍結 RM (Road Marking) 分支與鑑別器...")
        for p in model.decoder_rm.parameters():       p.requires_grad = False
        for p in model.discriminator_rm.parameters(): p.requires_grad = False
 
    if task_weights.get("ts", 1.0) == 0.0:
        print("  -> 凍結 TS (Traffic Sign) 分支與鑑別器...")
        for p in model.decoder_ts.parameters():       p.requires_grad = False
        for p in model.discriminator_ts.parameters(): p.requires_grad = False
 
    if task_weights.get("tl", 1.0) == 0.0:
        print("  -> 凍結 TL (Traffic Light) 分支與鑑別器...")
        for p in model.decoder_tl.parameters():       p.requires_grad = False
        for p in model.discriminator_tl.parameters(): p.requires_grad = False
    
    if task_weights.get("ll", 1.0) == 0.0:
        print("  -> 凍結 LL (Lane Line) 分支與鑑別器...")
        for p in model.decoder_ll.parameters():       p.requires_grad = False
        for p in model.discriminator_ll.parameters(): p.requires_grad = False

    ema.to(device)
 
    start_iter    = 0
    best_miou_sum = 0.0
 
    if checkpoint:
        print(f"Loading checkpoint from {checkpoint}...")
        ckpt     = torch.load(checkpoint, map_location=device)
        ckpt_sd  = ckpt["model_state_dict"]
        model_sd = model.state_dict()
        filtered = {k: v for k, v in ckpt_sd.items() if k in model_sd and v.shape == model_sd[k].shape}
        n_skip   = len(ckpt_sd) - len(filtered)
        print(f"  -> 跳過 {n_skip} 個 shape 不相容的 key（LL decoder 將重新初始化）")
        missing, _ = model.load_state_dict(filtered, strict=False)
        print(f"  -> 模型載入完成。Missing keys (預期為新 Head 的參數): {len(missing)}")
 
        if "ema" in ckpt:
            ema_ckpt_sd  = ckpt["ema"]
            ema_sd       = ema.state_dict()
            filtered_ema = {k: v for k, v in ema_ckpt_sd.items() if k in ema_sd and v.shape == ema_sd[k].shape}
            n_ema_skip   = len(ema_ckpt_sd) - len(filtered_ema)
            print(f"  -> EMA 載入: 跳過 {n_ema_skip} 個 shape 不相容的 key")
            ema.load_state_dict(filtered_ema, strict=False)
 
        best_miou_sum = ckpt.get("best_miou_sum", 0.0)
        if "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt:
            try:
                optimizer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                start_iter = ckpt.get("iteration", 0)
                print(f"  -> True Resume 模式：從 iter {start_iter + 1} 繼續訓練。")
            except Exception as e:
                start_iter = 0
                print(f"  -> optimizer/scheduler 載入失敗（{e}），fallback 到 Fine-tuning 模式（iter 歸零）。")
        else:
            start_iter = 0
            print(f"  -> 進入微調模式 (Fine-tuning)：checkpoint 無 optimizer 狀態，Iteration 歸零。")
 
    elif cfg.pretrain_path:
        ckpt = torch.load(cfg.pretrain_path)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        ema.ema_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded Pretrain. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
 
    loss_meter = {
        "rm_src": 0.0, "rm_tgt": 0.0, "rm_dis_s": 0.0, "rm_dis_t": 0.0,
        "ll_src": 0.0, "ll_tgt": 0.0, "ll_dis_s": 0.0, "ll_dis_t": 0.0,
        "ts_src": 0.0, "ts_tgt": 0.0, "ts_dis_s": 0.0, "ts_dis_t": 0.0,
        "tl_src": 0.0, "tl_tgt": 0.0, "tl_dis_s": 0.0, "tl_dis_t": 0.0,
    }
    batch_counter      = {"rm": 0, "ll": 0, "ts": 0, "tl": 0}
    recent_checkpoints = []
    rm_class_weights = torch.ones(len(cats_rm)).to(device)
    rm_class_weights[1:] = 5.0
    
    def _filter_pseudo_labels_by_geometry(pseudo_mask: torch.Tensor) -> torch.Tensor:
        """
        對 EMA 偽標籤做幾何濾波：
        用二次多項式擬合每個 connected component，
        丟棄殘差 > 15px 的幾何不合理區域（雨天反射常見的散點假陽性）。
        """
        result = pseudo_mask.clone()
        batch_np = pseudo_mask.cpu().numpy()

        for b in range(batch_np.shape[0]):
            fg = (batch_np[b] == 1.0).astype(np.uint8)
            if fg.sum() == 0:
                continue
            num_labels, labels = cv2.connectedComponents(fg)
            filtered = np.zeros_like(fg)

            for lbl in range(1, num_labels):
                ys, xs = np.where(labels == lbl)
                if len(ys) < 20:
                    continue
                try:
                    poly   = np.polyfit(ys, xs, deg=2)
                    xs_fit = np.polyval(poly, ys)
                    if len(ys) < 50:
                        if np.mean(np.abs(xs - xs_fit)) >= 15.0:
                            continue
                    else:
                        if np.mean(np.abs(xs - xs_fit)) >= 30.0:
                            continue
                    filtered[ys, xs] = 1
                except np.linalg.LinAlgError:
                    continue

            removed = (fg == 1) & (filtered == 0)
            result[b][torch.from_numpy(removed).to(result.device)] = 255.0

        return result


    def run_task_step(task_name, dl_src, dl_tgt, classmix, categories, _palette, metric, iter_idx):
        step_loss  = torch.tensor(0.0, device=device)
        disc_loss  = torch.tensor(0.0, device=device)
        src_data   = next(dl_src)
 
        if task_name in ["ts", "tl"]:
            imgs          = src_data["img"].to(device)
            raw_gt        = src_data["raw_gt"].to(device)
            batch_size    = imgs.shape[0]
            criterion_det = criterion_det_ts if task_name == "ts" else criterion_det_tl
            
            if task_name == "ts":
                _mu  = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1,3,1,1)
                _sig = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1,3,1,1)
                imgs_01  = (imgs * _sig + _mu).clamp(0, 1).cpu()
                fda_ts.beta = 0.005
                imgs_fda = torch.stack([
                    fda_ts.transform({"img": imgs_01[i]})["img"] for i in range(imgs_01.shape[0])
                ])
                imgs_fda      = (imgs_fda.to(device) - _mu) / _sig
                imgs_combined = torch.cat([imgs, imgs_fda], dim=0)
                raw_gt_combined = torch.cat([raw_gt, raw_gt], dim=0)
            else:
                imgs_combined   = imgs
                raw_gt_combined = raw_gt

            with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                out_src    = model(pixel_values=imgs_combined, task=task_name)
                outputs    = out_src["logits"]
                if task_name == "ts":
                    latent_src = tuple(h[:batch_size] for h in out_src["hidden_states"])
                else:
                    latent_src = out_src["hidden_states"]
                src_loss   = criterion_det(outputs, raw_gt_combined, use_reg_loss=True)
 
            if torch.isnan(src_loss) or torch.isinf(src_loss):
                print(f"[Warning] NaN/Inf {task_name.upper()} Loss at iter {iter_idx}")
                return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)
 
            loss_meter[f"{task_name}_src"] += src_loss.item()
            step_loss += src_loss

            _ADV_INTERVAL = 2
            if dl_tgt is not None and iter_idx % _ADV_INTERVAL == 0:
                tgt_data     = next(dl_tgt)
                tgt_imgs     = tgt_data["img"].to(device)
                WARMUP_ITERS = 1500
                if iter_idx > WARMUP_ITERS:
                    with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                        latent_tgt   = model.forward_encoder(tgt_imgs)
                        cur_h, cur_w = imgs.shape[2], imgs.shape[3]
                        p4_h, p4_w   = cur_h // 32, cur_w // 32
                        src_dis_lbl  = torch.zeros((batch_size,           1, p4_h, p4_w), device=device)
                        tgt_dis_lbl  = torch.ones ((tgt_imgs.shape[0],   1, p4_h, p4_w), device=device)
                        loss_d_src   = compute_domain_discrimination_loss_mt(
                            model, imgs, None, src_dis_lbl, None, (p4_h, p4_w), task_name, device, latent=latent_src,
                        )
                        loss_d_tgt   = compute_domain_discrimination_loss_mt(
                            model, tgt_imgs, None, tgt_dis_lbl, None, (p4_h, p4_w), task_name, device, latent=latent_tgt,
                        )
                        adv_weight = 0.2
                        if not (torch.isnan(loss_d_src) or torch.isinf(loss_d_src)):
                            disc_loss += loss_d_src * adv_weight
                            loss_meter[f"{task_name}_dis_s"] += loss_d_src.item()
                        if not (torch.isnan(loss_d_tgt) or torch.isinf(loss_d_tgt)):
                            disc_loss += loss_d_tgt * adv_weight
                            loss_meter[f"{task_name}_dis_t"] += loss_d_tgt.item()
 
            batch_counter[task_name] += 1
            return step_loss, disc_loss

        elif task_name == "ll":
            imgs       = src_data["img"].to(device)
            lane_mask  = src_data["lane_mask"].to(device)
            batch_size = imgs.shape[0]
 
            if lane_mask.shape[-2:] != imgs.shape[-2:]:
                lane_mask = F.interpolate(
                    lane_mask.unsqueeze(1).float(),
                    size=imgs.shape[-2:], mode="nearest",
                ).squeeze(1)
 
            with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                out_src    = model(pixel_values=imgs, task="ll")
                latent_src = out_src["hidden_states"]
                src_loss   = criterion_hybrid(out_src, lane_mask, is_target_domain=False)
 
            if torch.isnan(src_loss) or torch.isinf(src_loss):
                print(f"[Warning] NaN/Inf LL Loss at iter {iter_idx}")
                return torch.tensor(0.0, device=device, requires_grad=True), torch.tensor(0.0, device=device)
 
            loss_meter["ll_src"] += src_loss.item()
            step_loss += src_loss
 
            if dl_tgt is not None and iter_idx % ema.update_every == 0:
                tgt_data     = next(dl_tgt)
                tgt_imgs     = tgt_data["img"].to(device)
                WARMUP_ITERS = 1500
                if iter_idx > WARMUP_ITERS:
                    ema.ema_model.eval()
                    with torch.no_grad(), torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                        ema_out     = ema.ema_model(pixel_values=tgt_imgs, task="ll")
                        pseudo_prob = torch.sigmoid(ema_out["mask_logits"]).squeeze(1)
                        pseudo_mask = torch.zeros_like(pseudo_prob)
                        pseudo_mask[pseudo_prob > 0.65]                          = 1.0
                        pseudo_mask[(pseudo_prob >= 0.35) & (pseudo_prob < 0.65)] = 255.0
                    ema.ema_model.train()
                    pseudo_mask = _filter_pseudo_labels_by_geometry(pseudo_mask)

 
                    has_pseudo = (pseudo_mask == 1.0).any()
                    if has_pseudo:
                        with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                            out_tgt    = model(pixel_values=tgt_imgs, task="ll")
                            latent_tgt = out_tgt["hidden_states"]
                            unsup_loss = criterion_hybrid(out_tgt, pseudo_mask, is_target_domain=True)
                            if not (torch.isnan(unsup_loss) or torch.isinf(unsup_loss)):
                                step_loss += unsup_loss * 0.5
                                loss_meter["ll_tgt"] += unsup_loss.item()
                    else:
                        with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                            latent_tgt = model.forward_encoder(tgt_imgs)
 
                    with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                        cur_h, cur_w = imgs.shape[2], imgs.shape[3]
                        src_dis_lbl  = torch.zeros((batch_size, 1, cur_h, cur_w), device=device)
                        tgt_dis_lbl  = torch.ones((tgt_imgs.shape[0], 1, cur_h, cur_w), device=device)
                        loss_d_src   = compute_domain_discrimination_loss_mt(
                            model, imgs, None, src_dis_lbl, None, (cur_h, cur_w), "ll", device, latent=latent_src,
                        )
                        loss_d_tgt = compute_domain_discrimination_loss_mt(
                            model, tgt_imgs, None, tgt_dis_lbl, None, (cur_h, cur_w), "ll", device, latent=latent_tgt,
                        )
                        if not (torch.isnan(loss_d_src) or torch.isinf(loss_d_src)):
                            disc_loss += loss_d_src * 0.2
                            loss_meter["ll_dis_s"] += loss_d_src.item()
                        if not (torch.isnan(loss_d_tgt) or torch.isinf(loss_d_tgt)):
                            disc_loss += loss_d_tgt * 0.2
                            loss_meter["ll_dis_t"] += loss_d_tgt.item()
 
            batch_counter["ll"] += 1
            return step_loss, disc_loss

        elif task_name == "rm":
            src_imgs   = [im.to(device) for im in src_data["imgs" if "imgs" in src_data else "img"]]
            batch_size = src_imgs[0].shape[0]
            src_ann    = src_data["ann"].to(device)

            src_view_idx = 0

            with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                out_src    = model(pixel_values=src_imgs[src_view_idx], task=task_name)
                logits     = out_src["logits"]
                logits_aux = out_src.get("logits_aux")
                latent_src = out_src["hidden_states"]
                boundary_gt = get_boundary_map(src_ann, ignore_index=255).to(device)
                src_loss    = F.cross_entropy(logits.float(), src_ann, weight=rm_ce_weight,
                                              ignore_index=255) + 0.4 * dice_loss_rm_src(logits, src_ann)
                if logits_aux is not None:
                    src_ann_small = F.interpolate(src_ann.unsqueeze(1).float(), scale_factor=0.25, mode='nearest').squeeze(1).long()
                    src_loss = src_loss + 0.1 * F.cross_entropy(logits_aux.float(), src_ann_small, weight=rm_ce_weight, ignore_index=255)
                boundary_small = out_src.get("boundary_logits")
                if boundary_small is not None:
                    boundary_pred_full = F.interpolate(
                        torch.nan_to_num(boundary_small.float(), nan=0.0, posinf=50.0, neginf=-50.0),
                        size=boundary_gt.shape[-2:],
                        mode='bilinear', align_corners=False
                    ).squeeze(1)
                    bce_boundary = F.binary_cross_entropy_with_logits(
                        boundary_pred_full, boundary_gt,
                        pos_weight=torch.tensor(8.0, device=device)
                    )
                    if not (torch.isnan(bce_boundary) or torch.isinf(bce_boundary)):
                        src_loss = src_loss + 0.15 * bce_boundary
                if out_src.get("feat_small") is not None:
                    ann_small_proto = F.interpolate(
                        src_ann.unsqueeze(1).float(), scale_factor=0.25, mode='nearest'
                    ).squeeze(1).long()
                    with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                        hidden_det = [h.detach() for h in out_src["hidden_states"]]
                        _, _, feat_for_proto = model.decoder_rm(hidden_det)
                    loss_proto = proto_loss_rm(feat_for_proto.float(), ann_small_proto)
                    if not (torch.isnan(loss_proto) or torch.isinf(loss_proto)):
                        src_loss = src_loss + 0.1 * loss_proto

            if torch.isnan(src_loss) or torch.isinf(src_loss):
                print(f"[Warning] NaN/Inf RM Src Loss at iter {iter_idx}")
                src_loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                pred = logits.argmax(dim=1)
                metric.compute_and_accum(pred.cpu(), src_ann.cpu())
            loss_meter[f"{task_name}_src"] += src_loss.item()
            step_loss = src_loss
 
            if iter_idx % ema.update_every == 0 and iter_idx > 1500:
                tgt_data            = next(dl_tgt)
                tgt_imgs_cpu        = tgt_data["imgs" if "imgs" in tgt_data else "img"]
                erased_tgt_imgs_cpu = tgt_data.get("erased imgs", tgt_data.get("erased img", None))
 
                ema.ema_model.eval()
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                    tgt_imgs_dev = [im.to(device) for im in tgt_imgs_cpu]
                    tgt_view_idx = 0
                    ema_out      = ema.ema_model(pixel_values=tgt_imgs_dev[tgt_view_idx], task=task_name)
                    tgt_prob     = ema_out["logits"].softmax(1)
                ema.ema_model.train()
 
                tgt_prob  = tgt_prob.cpu().detach()
                dis_label = torch.ones((batch_size, 1, cfg.crop_size[0], cfg.crop_size[1]))

                tgt_imgs_mixed = [im.clone() for im in tgt_imgs_cpu]
                classmix.mix(
                    mix_domain=(0 if random.random() > 0.5 else 1),
                    imgs=tgt_imgs_mixed,
                    ann=tgt_prob,
                    task=task_name,
                    dis_label=dis_label,
                    erased_imgs=erased_tgt_imgs_cpu,
                    class_thresholds=rm_class_thresholds,
                )
 
                with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                    tgt_imgs_mixed_dev = [im.to(device) for im in tgt_imgs_mixed]
                    tgt_prob_dev       = tgt_prob.to(device)
                    out_tgt            = model(pixel_values=tgt_imgs_mixed_dev[tgt_view_idx], task=task_name)
                    latent_tgt         = out_tgt["hidden_states"]
                    unsup_loss         = target_criterion.compute(
                        out_tgt["logits"], tgt_prob_dev,
                        class_weights=rm_class_weights,
                        class_thresholds=rm_class_thresholds,
                    )
                    if not (torch.isnan(unsup_loss) or torch.isinf(unsup_loss)):
                        step_loss = step_loss + unsup_loss * 0.5
                        loss_meter[f"{task_name}_tgt"] += unsup_loss.item()

                    if erased_tgt_imgs_cpu is not None:
                        erased_imgs_dev = [im.to(device) for im in erased_tgt_imgs_cpu]
                        out_erase       = model(pixel_values=erased_imgs_dev[tgt_view_idx], task=task_name)
                        loss_erase      = target_criterion.compute(
                            out_erase["logits"], tgt_prob_dev,
                            class_weights=rm_class_weights,
                            class_thresholds=rm_class_thresholds,
                        )
                        if not (torch.isnan(loss_erase) or torch.isinf(loss_erase)):
                            step_loss = step_loss + loss_erase * 0.5
 
                with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                    src_dis_lbl = torch.zeros((batch_size, 1, cfg.crop_size[0], cfg.crop_size[1])).to(device)
                    loss_d_src  = compute_domain_discrimination_loss_mt(
                        model, src_imgs, None, src_dis_lbl, None, cfg.crop_size, task_name, device,
                        latent=latent_src,
                    )
                    if not (torch.isnan(loss_d_src) or torch.isinf(loss_d_src)):
                        disc_loss += loss_d_src * 0.2
                        loss_meter[f"{task_name}_dis_s"] += loss_d_src.item()

                with torch.amp.autocast("cuda", enabled=cfg.autocast, dtype=torch.bfloat16):
                    tgt_dis_lbl = dis_label.to(device)
                    loss_d_tgt  = compute_domain_discrimination_loss_mt(
                        model, tgt_imgs_mixed_dev, None, tgt_dis_lbl, None, cfg.crop_size, task_name, device,
                        latent=latent_tgt,
                    )
                    if not (torch.isnan(loss_d_tgt) or torch.isinf(loss_d_tgt)):
                        disc_loss += loss_d_tgt * 0.2
                        loss_meter[f"{task_name}_dis_t"] += loss_d_tgt.item()
 
            batch_counter[task_name] += 1
            return step_loss, disc_loss

    milestones       = np.linspace(0, cfg.max_iters, len(cfg.ema_update_intervals) + 1)[1:].astype(int)
    total_start_time = time.time()
 
    for iter_idx in range(start_iter + 1, cfg.max_iters + 1):
        for idx, ms in enumerate(milestones):
            if iter_idx == ms:
                ema.update_every = cfg.ema_update_intervals[min(idx + 1, len(cfg.ema_update_intervals) - 1)]
                break
            elif iter_idx < milestones[0]:
                ema.update_every = cfg.ema_update_intervals[0]
 
        model.train()
        optimizer.zero_grad()
 
        task_weights = cfg.task_weight if isinstance(cfg.task_weight, dict) else vars(cfg.task_weight)
        ll_interval  = int(getattr(cfg, 'll_interval',  2))
        ts_interval  = int(getattr(cfg, 'ts_interval',  3))
        tl_interval  = int(getattr(cfg, 'tl_interval',  6))

        if task_weights.get("rm", 1.0) > 0.0:
            task_loss_rm, disc_loss_rm = run_task_step("rm", dl_rm_src, dl_rm_tgt, classmix_rm, cats_rm, palette_rm, metric_rm, iter_idx)
            scaler.scale(mt_loss_wrapper.get_weighted_loss(task_loss_rm, 0) + disc_loss_rm).backward()

        if task_weights.get("ll", 1.0) > 0.0 and iter_idx % ll_interval == 0:
            task_loss_ll, disc_loss_ll = run_task_step("ll", dl_ll_src, dl_ll_tgt, None, None, None, None, iter_idx)
            scaler.scale(mt_loss_wrapper.get_weighted_loss(task_loss_ll, 1) + disc_loss_ll).backward()

        if task_weights.get("ts", 1.0) > 0.0 and iter_idx % ts_interval == 0:
            task_loss_ts, disc_loss_ts = run_task_step("ts", dl_ts_src, dl_ts_tgt, None, None, None, None, iter_idx)
            scaler.scale(mt_loss_wrapper.get_weighted_loss(task_loss_ts, 2) + disc_loss_ts).backward()

        if task_weights.get("tl", 1.0) > 0.0 and iter_idx % tl_interval == 0:
            task_loss_tl, disc_loss_tl = run_task_step("tl", dl_tl_src, dl_tl_tgt, None, None, None, None, iter_idx)
            scaler.scale(mt_loss_wrapper.get_weighted_loss(task_loss_tl, 3) + disc_loss_tl).backward()
        
        scaler.unscale_(optimizer)
        all_params = list(model.parameters()) + list(mt_loss_wrapper.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update()
        scheduler.step()
 
        if iter_idx % cfg.train_interval == 0:
            with torch.no_grad():
                log_vars = mt_loss_wrapper.log_vars
                weights = torch.exp(-log_vars)
                w_rm, w_ll, w_ts, w_tl = [weights[i].item() for i in range(4)]
 
            denom_rm = batch_counter["rm"] if batch_counter["rm"] > 0 else 1
            denom_ll = batch_counter["ll"] if batch_counter["ll"] > 0 else 1
            denom_ts = batch_counter["ts"] if batch_counter["ts"] > 0 else 1
            denom_tl = batch_counter["tl"] if batch_counter["tl"] > 0 else 1
 
            rm_res  = metric_rm.get_and_reset()["IoU"]
            rm_miou = np.nanmean(rm_res) if len(rm_res) > 0 else 0.0
 
            print(f"[Iter {iter_idx}] LR: {optimizer.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  [AutoWeights] RM: {w_rm:.4f} | LL: {w_ll:.4f} | TS: {w_ts:.4f} | TL: {w_tl:.4f}")
            print(f"  RM | Loss Src: {loss_meter['rm_src']/denom_rm:.4f} Tgt: {loss_meter['rm_tgt']/denom_rm:.4f} Dis: {loss_meter['rm_dis_s']/denom_rm:.4f} mIoU: {rm_miou:.4f}")
            print(f"  LL | Loss Src: {loss_meter['ll_src']/denom_ll:.4f} Tgt: {loss_meter['ll_tgt']/denom_ll:.4f} Dis: {loss_meter['ll_dis_s']/denom_ll:.4f}")
            print(f"  TS | Loss Src: {loss_meter['ts_src']/denom_ts:.4f} Tgt: {loss_meter['ts_tgt']/denom_ts:.4f} Dis: {loss_meter['ts_dis_s']/denom_ts:.4f}")
            print(f"  TL | Loss Src: {loss_meter['tl_src']/denom_tl:.4f} Tgt: {loss_meter['tl_tgt']/denom_tl:.4f} Dis: {loss_meter['tl_dis_s']/denom_tl:.4f}")
 
            writer.add_scalar("RM/Loss_Source", loss_meter["rm_src"] / denom_rm, iter_idx)
            writer.add_scalar("RM/mIoU_Train",  rm_miou,                         iter_idx)
            writer.add_scalar("LL/Loss_Source", loss_meter["ll_src"] / denom_ll, iter_idx)
            writer.add_scalar("TS/Loss_Source", loss_meter["ts_src"] / denom_ts, iter_idx)
            writer.add_scalar("TL/Loss_Source", loss_meter["tl_src"] / denom_tl, iter_idx)
 
            for k in loss_meter: loss_meter[k] = 0.0
            batch_counter["rm"] = batch_counter["ll"] = batch_counter["ts"] = batch_counter["tl"] = 0
 
        if iter_idx % cfg.val_interval == 0:
            print("Running Validation...")
            model.eval()

            def validate_task(name, validator_src, validator_tgt):
                with torch.no_grad(), torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                    _, m_s, _, _ = validator_src.validate()
                    _, m_t, _, _ = validator_tgt.validate()
                writer.add_scalar(f"{name.upper()}/Val_Source_mIoU", m_s, iter_idx)
                writer.add_scalar(f"{name.upper()}/Val_Target_mIoU", m_t, iter_idx)
                print(f"  {name.upper()} Val | Src mIoU: {m_s:.4f}, Tgt mIoU: {m_t:.4f}")
                return m_s, m_t
 
            rm_s_iou, rm_t_iou = validate_task("rm", validator_rm_src, validator_rm_tgt)
 
            ll_val_score = 0.0
            ll_tgt_score = 0.0
            if dl_ll_src_val is not None:
                ll_val_score = validate_ll_task(model, dl_ll_src_val, None, device)
                print(f"  LL Val | IoU: {ll_val_score:.4f}")
                writer.add_scalar("LL/Val_IoU", ll_val_score, iter_idx)
            if dl_ll_tgt_val is not None:
                ll_tgt_score = validate_ll_task(model, dl_ll_tgt_val, None, device)
                print(f"  LL Val | Tgt IoU: {ll_tgt_score:.4f}")
                writer.add_scalar("LL/Val_Target_IoU", ll_tgt_score, iter_idx)
 
            ts_src_map = 0.0
            ts_tgt_map = 0.0
            if dl_ts_src_val is not None:
                ts_src_map = validate_det_task(model, dl_ts_src_val, device, cfg.ts_num_classes)
                print(f"  TS Val | Src mAP: {ts_src_map:.4f}")
                writer.add_scalar("TS/Val_Source_mAP", ts_src_map, iter_idx)
            if dl_ts_tgt_val is not None:
                ts_tgt_map = validate_det_task(model, dl_ts_tgt_val, device, cfg.ts_num_classes)
                print(f"  TS Val | Tgt mAP: {ts_tgt_map:.4f}")
                writer.add_scalar("TS/Val_Target_mAP", ts_tgt_map, iter_idx)

            tl_src_map = 0.0
            tl_tgt_map = 0.0
            if dl_tl_src_val is not None:
                tl_src_map = validate_det_task(model, dl_tl_src_val, device, cfg.tl_num_classes, task="tl", nms_thresh=0.35)
                print(f"  TL Val | Src mAP: {tl_src_map:.4f}")
                writer.add_scalar("TL/Val_Source_mAP", tl_src_map, iter_idx)
            if dl_tl_tgt_val is not None:
                tl_tgt_map = validate_det_task(model, dl_tl_tgt_val, device, cfg.tl_num_classes, task="tl", nms_thresh=0.35)
                print(f"  TL Val | Tgt mAP: {tl_tgt_map:.4f}")
                writer.add_scalar("TL/Val_Target_mAP", tl_tgt_map, iter_idx)
 
            print("Visualizing predictions...")
            with torch.no_grad():
                visualize_validation_samples(
                    model, dl_rm_tgt_val, palette_rm, "rm", device, writer, iter_idx,
                    num_samples=3, title_prefix="Target_Pred_Mask",
                )
 
            print("Visualizing LL predictions...")
            with torch.no_grad():
                visualize_lstr_predictions(
                    model,
                    dl_ll_src_val if dl_ll_src_val else dl_ll_src,
                    device, writer, iter_idx, num_samples=3, title="LL_Curve_Src",
                )
 
            vis_ts = dl_ts_tgt_val if dl_ts_tgt_val else (dl_ts_src_val or dl_ts_src)
            print("Visualizing TS predictions...")
            with torch.no_grad():
                visualize_ts_predictions(
                    model, vis_ts, device, writer, iter_idx,
                    num_samples=3, confidence_threshold=0.3, title="TS_Pred_BBox", task="ts",
                )
 
            vis_tl = dl_tl_tgt_val if dl_tl_tgt_val else (dl_tl_src_val or dl_tl_src)
            print("Visualizing TL predictions...")
            with torch.no_grad():
                visualize_ts_predictions(
                    model, vis_tl, device, writer, iter_idx,
                    num_samples=3, confidence_threshold=0.3, title="TL_Pred_BBox", task="tl",
                )

            current_total_miou = rm_s_iou + rm_t_iou + ll_val_score + ll_tgt_score + ts_src_map + ts_tgt_map + tl_src_map + tl_tgt_map
            ckpt_content = {
                "model_state_dict":     model.state_dict(),
                "ema":                  ema.state_dict(),
                "optimizer_state_dict": optimizer.optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "iteration":            iter_idx,
                "best_miou_sum":        max(best_miou_sum, current_total_miou),
            }
 
            torch.save(ckpt_content, f"{tb_dir}/latest_model.pth")
            current_ckpt_path = f"{tb_dir}/checkpoint_iter{iter_idx}.pth"
            torch.save(ckpt_content, current_ckpt_path)
            recent_checkpoints.append(current_ckpt_path)
 
            if len(recent_checkpoints) > 3:
                oldest = recent_checkpoints.pop(0)
                oldest_iter = int(oldest.split("iter")[-1].replace(".pth", ""))
                if oldest_iter in cfg.milestone_iters:
                    print(f"  [Storage] Kept milestone checkpoint: {os.path.basename(oldest)}")
                elif os.path.exists(oldest):
                    os.remove(oldest)
                    print(f"  [Storage] Removed old checkpoint: {os.path.basename(oldest)}")
 
            if current_total_miou > best_miou_sum:
                best_miou_sum = current_total_miou
                torch.save(ckpt_content, f"{tb_dir}/best_model.pth")
                print(f"  New Best Model! Total Score: {best_miou_sum:.4f}")
 
            gc.collect()
 
    total_duration    = time.time() - total_start_time
    duration_str      = str(timedelta(seconds=int(total_duration)))
    max_mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    max_mem_reserved  = torch.cuda.max_memory_reserved(device)  / (1024 ** 3)
 
    print("\n" + "=" * 50)
    print("Training Completed Successfully!")
    print(f"Total Training Time : {duration_str}")
    print(f"Peak VRAM Allocated : {max_mem_allocated:.2f} GB (模型與資料真實最高佔用)")
    print(f"Peak VRAM Reserved  : {max_mem_reserved:.2f} GB (PyTorch 總共圈佔的最高記憶體)")
    print("=" * 30 + "\n")
    writer.close()
 
 
if __name__ == "__main__":
    assert len(sys.argv) >= 2
    cfg        = MultiTaskTrainingConfig.load(sys.argv[1])
    exp_name   = sys.argv[1].split("/")[-1].replace(".json", "")
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else None
    log_dir    = None if checkpoint is None else os.path.dirname(checkpoint).split("/")[-1]
    main(cfg, exp_name, checkpoint, log_dir)
 