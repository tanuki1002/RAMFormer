# find_rm_val_picture.py
# 對指定資料夾中的每張圖片，計算單張圖片的 RM 任務 mIoU，
# 找出表現最好的 Top-K 張圖片，輸出圖片名稱與視覺化結果。
#
# 用法範例：
#   python tools/find_rm_val_picture.py \
#       --checkpoint logs/my_exp/best_model.pth \
#       --config configs/train_uda_multi_tasks.json \
#       --img_dir /path/to/val/images \
#       --ann_dir /path/to/val/labels \
#       --top_k 10 --output top_rm_images
#
# 參數說明：
#   --checkpoint : .pth checkpoint 路徑
#   --config     : 訓練設定的 JSON 路徑
#   --img_dir    : 圖片資料夾路徑
#   --ann_dir    : 標籤資料夾路徑
#   --top_k      : 輸出 mIoU 最高的幾張（預設 10）
#   --output     : 視覺化圖片的輸出目錄（預設 top_rm_images）
#   --no_vis     : 加上此旗標則只輸出名稱，不存視覺化圖片

import argparse
import os
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from transformers import SegformerConfig
from tqdm import tqdm

from engine.multi_task_segformer import get_model as get_mt_model
from engine.dataloader import get_dataset
from engine.category import Category
from engine import transform
from engine.metric import Metrics
from engine.validator import Validator
from configs.multitask_config import MultiTaskTrainingConfig


# =============================================================================
# Utility（與 evaluate_all_tasks.py 保持一致）
# =============================================================================

class Cleanup:
    def transform(self, data):
        keys = list(data.keys())
        for k in keys:
            if data[k] is None:
                del data[k]
        return data


class TaskModelWrapper(torch.nn.Module):
    def __init__(self, model, task):
        super().__init__()
        self.model = model
        self.task = task

    def forward(self, images=None, pixel_values=None, **kwargs):
        img_input = pixel_values if pixel_values is not None else images
        if isinstance(img_input, list):
            img_input = img_input[0]
        kwargs.pop('task', None)
        outputs = self.model(pixel_values=img_input, task=self.task, **kwargs)
        if isinstance(outputs, dict):
            logits = outputs.get('mask_logits', outputs.get('logits'))
            if logits is not None and logits.shape[-2:] != img_input.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=img_input.shape[-2:], mode='bilinear', align_corners=False
                )
            return logits, outputs.get('loss', None)
        return outputs


def rm_post_process(pred_mask: np.ndarray, min_area: int = 400) -> np.ndarray:
    result = pred_mask.copy()
    fg = (pred_mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        mask = labels == lab
        if area < min_area:
            result[mask] = 0
            continue
        classes = pred_mask[mask]
        fg_ids = classes[classes > 0]
        if len(fg_ids) == 0:
            continue
        vals, cnts = np.unique(fg_ids, return_counts=True)
        result[mask] = vals[cnts.argmax()]
    return result


def get_palette_from_cats(categories):
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i, c in enumerate(categories):
        if i < 256:
            palette[i] = [c.r, c.g, c.b]
    return palette


# =============================================================================
# 計算單張圖片的 per-class IoU，並回傳平均（mIoU，只計算 GT 中出現的 class）
# =============================================================================

def compute_per_image_miou(pred: np.ndarray, gt: np.ndarray,
                           num_classes: int, ignore_ids: list) -> float:
    """
    pred, gt : HxW numpy uint8 arrays（尺寸需相同）
    回傳該張圖片的 mIoU（只對 GT 中出現的非 ignore 類別取平均）。
    """
    present_classes = np.unique(gt).tolist()
    iou_sum = 0.0
    valid_count = 0

    for c in range(num_classes):
        if c in ignore_ids:
            continue
        if c not in present_classes:
            continue  # GT 中沒有此類別，跳過
        pred_c = (pred == c)
        gt_c   = (gt == c)
        inter  = np.logical_and(pred_c, gt_c).sum()
        union  = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        iou_sum += inter / union
        valid_count += 1

    return iou_sum / valid_count if valid_count > 0 else 0.0


# =============================================================================
# 儲存單張圖片的視覺化結果（GT / Prediction / Error Map）
# =============================================================================

def save_rm_vis_single(img_bgr, pred_np, gt_np, palette, save_path):
    orig_h, orig_w = img_bgr.shape[:2]

    pred_color_bgr = cv2.cvtColor(palette[pred_np], cv2.COLOR_RGB2BGR)
    gt_color_bgr   = cv2.cvtColor(palette[gt_np],   cv2.COLOR_RGB2BGR)

    error_map = np.zeros_like(img_bgr)
    valid_mask   = (gt_np != 255) & ((gt_np != 0) | (pred_np != 0))
    correct_mask = (gt_np == pred_np) & valid_mask
    wrong_mask   = (gt_np != pred_np) & valid_mask & (gt_np != 255)
    error_map[correct_mask] = [0, 255, 0]   # 綠：正確
    error_map[wrong_mask]   = [0, 0, 255]   # 紅：錯誤

    alpha = 0.6
    vis_gt   = img_bgr.copy()
    vis_pred = img_bgr.copy()
    vis_err  = img_bgr.copy()

    gt_fg = gt_np > 0
    if gt_fg.any():
        vis_gt[gt_fg] = cv2.addWeighted(img_bgr[gt_fg], 1 - alpha, gt_color_bgr[gt_fg], alpha, 0)

    pred_fg = pred_np > 0
    if pred_fg.any():
        vis_pred[pred_fg] = cv2.addWeighted(img_bgr[pred_fg], 1 - alpha, pred_color_bgr[pred_fg], alpha, 0)

    err_fg = correct_mask | wrong_mask
    if err_fg.any():
        vis_err[err_fg] = cv2.addWeighted(img_bgr[err_fg], 0.3, error_map[err_fg], 0.7, 0)

    final_vis = np.hstack([vis_gt, vis_pred, vis_err])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_vis, "1. Ground Truth", (10, 40), font, 0.8, (0, 0, 255), 2)
    cv2.putText(final_vis, "2. Prediction",   (orig_w + 10, 40), font, 0.8, (0, 255, 0), 2)
    cv2.putText(final_vis, "3. Error Map (Green=OK, Red=Wrong)",
                (orig_w * 2 + 10, 40), font, 0.8, (0, 255, 255), 2)

    cv2.imwrite(save_path, final_vis)


# =============================================================================
# 主程式
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Per-image RM mIoU analysis: find the best images in a folder."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to training config JSON")
    parser.add_argument("--img_dir",    type=str, required=True,
                        help="Folder containing input images")
    parser.add_argument("--ann_dir",    type=str, required=True,
                        help="Folder containing annotation labels")
    parser.add_argument("--top_k",      type=int, default=10,
                        help="Number of top images to report (default: 10)")
    parser.add_argument("--output",     type=str, default="top_rm_images",
                        help="Output directory for visualization images (default: top_rm_images)")
    parser.add_argument("--no_vis",     action="store_true",
                        help="Print filenames only, do not save visualization images")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. 載入設定 ──────────────────────────────────────────────────────
    print(f"Loading config: {args.config}")
    cfg = MultiTaskTrainingConfig.load(args.config)
    cats_rm = Category.load(cfg.category_csv_rlmd)
    num_classes = len(cats_rm)
    palette = get_palette_from_cats(cats_rm)

    _raw_ignore = getattr(cfg, 'ignore_index', [[255]])
    class_ignore_rm = _raw_ignore[0] if _raw_ignore else [255]
    if 255 not in class_ignore_rm:
        class_ignore_rm = list(class_ignore_rm) + [255]

    # ── 2. 建立 DataLoader（batch_size=1 以便 per-image 評估）────────────
    transforms_rm_val = [
        transform.LoadImg(),
        transform.LoadAnn(),
        transform.ToTensor(),
        transform.Resize(cfg.image_scale),
        transform.Normalize(),
        transform.Check(),
        Cleanup()
    ]

    ds = get_dataset(cfg.dataset_rlmd, args.img_dir, args.ann_dir,
                     None, transforms_rm_val)
    if ds is None or len(ds) == 0:
        print(f"Error: No data found at img_dir={args.img_dir}")
        return
    print(f"Dataset: {len(ds)} images  |  Device: {device}")

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1,
                    pin_memory=cfg.pin_memory)

    # ── 3. 建立模型並載入權重 ─────────────────────────────────────────────
    print("Building model and loading checkpoint...")
    config_rm = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_rm.num_labels = num_classes
    config_ll = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_ll.num_labels = len(Category.load(cfg.category_csv_ll))
    config_ts = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_ts.num_labels = getattr(cfg, 'ts_num_classes', 13)
    config_tl = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_tl.num_labels = getattr(cfg, 'tl_num_classes', 4)

    model = get_mt_model(config_rm, config_ll, config_ts, config_tl)
    model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'ema' in ckpt:
        ema_data = ckpt['ema']
        state_dict = ema_data.get('ema_model', ema_data)
    else:
        state_dict = ckpt

    clean_sd = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue
        name = k.replace("module.", "")
        if "discriminator" in name:
            continue
        clean_sd[name] = v

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"  [Checkpoint] Missing keys: {len(missing)}")
    model.eval()

    # ── 4. 建立 Validator（用於 slide inference）────────────────────────
    # TaskModelWrapper 讓 Validator 可以正確呼叫 forward with task='rm'
    model_wrapper = TaskModelWrapper(model, task='rm')

    # ── 5. Per-image mIoU 評估迴圈 ────────────────────────────────────
    print("\nComputing per-image mIoU...")
    records = []  # [(img_path, miou, pred_np, gt_np)]

    metric = Metrics(num_categories=num_classes, ignore_ids=class_ignore_rm, nan_to_num=0)
    # Validator 的 slide_inference 需要 dataloader，這裡改為手動呼叫 slide_inference
    # 直接建立 Validator 但不呼叫 .validate()，改用它的 slide_inference 方法
    validator = Validator(
        dataloader=dl,
        model=model_wrapper,
        device=device,
        metric=metric,
        crop_size=cfg.crop_size,
        stride=cfg.stride,
        num_classes=num_classes,
        mode='slide',
        ignore_index=class_ignore_rm,
    )

    autocast_enabled = (device.type == 'cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=autocast_enabled):
        for data in tqdm(dl, desc="Evaluating", dynamic_ncols=True):
            imgs = [data['img'].to(device)]
            ann  = data['ann'].to(device)        # shape: [1, H, W]
            img_path = data['img_path'][0]

            # slide_inference 回傳 [1, num_classes, H, W] logits（model 輸入尺寸）
            logits = validator.slide_inference(images=imgs)
            pred   = logits.argmax(dim=1)        # [1, H, W]

            # 將 prediction 與 GT 縮到相同大小（model 輸入尺寸）後轉成 numpy
            if pred.shape[-2:] != ann.shape[-2:]:
                ann_for_eval = torch.nn.functional.interpolate(
                    ann.unsqueeze(1).float(), size=pred.shape[-2:], mode='nearest'
                ).squeeze(1).long()
            else:
                ann_for_eval = ann

            pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
            gt_np   = ann_for_eval.squeeze(0).cpu().numpy().astype(np.uint8)

            miou = compute_per_image_miou(pred_np, gt_np, num_classes, class_ignore_rm)
            records.append((img_path, miou, pred_np, gt_np))

    # ── 6. 排序並輸出 Top-K 結果 ────────────────────────────────────────
    records.sort(key=lambda x: x[1], reverse=True)
    top_records = records[:args.top_k]

    print(f"\n{'TOP ' + str(args.top_k) + ' IMAGES (RM mIoU)':^60}")
    print("=" * 60)
    print(f"{'Rank':<6} {'mIoU':>8}  {'Filename'}")
    print("-" * 60)
    for i, (img_path, miou, _, _) in enumerate(top_records):
        fname = os.path.basename(img_path)
        print(f"{i+1:<6} {miou:>8.4f}  {fname}")
    print("=" * 60)

    # ── 7. 儲存結果名稱至 txt ─────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    txt_path = os.path.join(args.output, "top_images.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Top {args.top_k} images by RM per-image mIoU\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"img_dir:    {args.img_dir}\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Rank':<6} {'mIoU':>8}  Filename\n")
        f.write("-" * 60 + "\n")
        for i, (img_path, miou, _, _) in enumerate(top_records):
            fname = os.path.basename(img_path)
            f.write(f"{i+1:<6} {miou:>8.4f}  {fname}\n")
        f.write("=" * 60 + "\n")
    print(f"\nFilename list saved: {txt_path}")

    # ── 8. 儲存視覺化圖片 ─────────────────────────────────────────────
    if not args.no_vis:
        print(f"\nSaving visualization images to: {args.output}/")
        for i, (img_path, miou, pred_np, gt_np) in enumerate(top_records):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  [Skip] Cannot read: {img_path}")
                continue

            orig_h, orig_w = img_bgr.shape[:2]

            # pred / gt 來自 model 輸入尺寸，需放大回原圖尺寸
            pred_orig = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            gt_orig   = cv2.resize(gt_np,   (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            pred_orig = rm_post_process(pred_orig, min_area=int(orig_h * orig_w * 0.0003))

            fname_stem = os.path.splitext(os.path.basename(img_path))[0]
            save_path  = os.path.join(args.output, f"rank{i+1:02d}_miou{miou:.4f}_{fname_stem}.jpg")
            save_rm_vis_single(img_bgr, pred_orig, gt_orig, palette, save_path)
            print(f"  [{i+1:02d}] mIoU={miou:.4f}  {os.path.basename(save_path)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
