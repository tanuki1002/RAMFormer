# find_ll_val_picture.py
# 對指定資料夾中的每張圖片，計算單張圖片的 LL 任務 F1@0.5，
# 找出表現最好的 Top-K 張圖片，輸出 GT 與預測結果的對比視覺化。
#
# 用法範例：
#   python tools/find_ll_val_picture.py \
#       --checkpoint logs/my_exp/best_model.pth \
#       --config     configs/train_uda_multi_tasks.json \
#       --img_dir    /path/to/val/images \
#       --ann_dir    /path/to/val/lane_mask \
#       --top_k 10 --output top_ll_images
#
# 參數說明：
#   --checkpoint  : .pth checkpoint 路徑
#   --config      : 訓練設定的 JSON 路徑
#   --img_dir     : 圖片資料夾路徑
#   --ann_dir     : 標籤資料夾路徑（BDD100K binary PNG masks）
#   --top_k       : 輸出 F1@0.5 最高的幾張（預設 10）
#   --output      : 視覺化圖片的輸出目錄（預設 top_ll_images）
#   --prob_thresh : sigmoid 二值化閾值（預設 0.3）
#   --no_vis      : 加上此旗標則只輸出名稱清單，不存視覺化圖片

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import SegformerConfig

# 從 evaluate_all_tasks.py 取用 LL 評估函式（避免重複實作）
_tools_dir = os.path.dirname(os.path.abspath(__file__))
if _tools_dir not in sys.path:
    sys.path.insert(0, _tools_dir)
from evaluate_all_tasks import get_pred_30px_regions, get_gt_30px_regions

from configs.multitask_config import MultiTaskTrainingConfig
from engine import transform
from engine.category import Category
from engine.dataloader import get_dataset
from engine.multi_task_segformer import get_model as get_mt_model


# =============================================================================
# Utility
# =============================================================================

class Cleanup:
    def transform(self, data):
        keys = list(data.keys())
        for k in keys:
            if data[k] is None:
                del data[k]
        return data


# =============================================================================
# Per-image F1@0.5 計算
# =============================================================================

def compute_per_image_ll_f1(logit_np, gt_np, prob_thresh=0.3, lane_width=30, iou_thresh=0.5):
    """
    logit_np : [H, W] float numpy，模型原始 logit
    gt_np    : [H, W] float numpy，0=背景, 1=車道線
    回傳 dict：f1, precision, recall, pixel_iou, tp, fp, fn, n_pred, n_gt
    """
    p_prob = 1.0 / (1.0 + np.exp(-logit_np.astype(np.float64)))
    p_bin  = (p_prob > prob_thresh).astype(np.uint8)
    g_bin  = (gt_np  > 0.5).astype(np.uint8)

    # Pixel IoU
    inter     = np.logical_and(p_bin, g_bin).sum()
    union     = np.logical_or(p_bin, g_bin).sum()
    pixel_iou = float(inter) / (float(union) + 1e-8)

    p_regions = get_pred_30px_regions(p_bin, lane_width=lane_width)
    g_regions = get_gt_30px_regions(g_bin,  lane_width=lane_width, min_pixels=15)

    n_pred, n_gt = len(p_regions), len(g_regions)
    eps = 1e-8

    if n_pred == 0 and n_gt == 0:
        return dict(f1=1.0, precision=1.0, recall=1.0,
                    pixel_iou=pixel_iou, tp=0, fp=0, fn=0, n_pred=0, n_gt=0)
    if n_pred == 0:
        return dict(f1=0.0, precision=1.0, recall=0.0,
                    pixel_iou=pixel_iou, tp=0, fp=0, fn=n_gt, n_pred=0, n_gt=n_gt)
    if n_gt == 0:
        return dict(f1=0.0, precision=0.0, recall=1.0,
                    pixel_iou=pixel_iou, tp=0, fp=n_pred, fn=0, n_pred=n_pred, n_gt=0)

    # IoU matrix
    iou_mat = np.zeros((n_pred, n_gt))
    for i, pm in enumerate(p_regions):
        for j, gm in enumerate(g_regions):
            inter_r = np.logical_and(pm, gm).sum()
            union_r = np.logical_or(pm, gm).sum()
            iou_mat[i, j] = inter_r / (union_r + 1e-8)

    # 依信心分數排序（高信心優先匹配）
    if n_pred > 1:
        p_conf  = np.array([float(p_prob[pm == 1].mean()) if pm.sum() > 0 else 0.0
                            for pm in p_regions])
        iou_mat = iou_mat[np.argsort(p_conf)[::-1]]

    # Greedy matching
    tp, matched_gt = 0, set()
    for i in range(n_pred):
        best_g, best_iou = -1, -1.0
        for j in range(n_gt):
            if j in matched_gt:
                continue
            if iou_mat[i, j] > best_iou:
                best_iou, best_g = iou_mat[i, j], j
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_g)

    fp        = n_pred - tp
    fn        = n_gt   - len(matched_gt)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    return dict(f1=float(f1), precision=float(precision), recall=float(recall),
                pixel_iou=pixel_iou, tp=tp, fp=fp, fn=fn, n_pred=n_pred, n_gt=n_gt)


# =============================================================================
# 視覺化
# =============================================================================

def _overlay_regions(canvas, regions, color_bgr, alpha=0.55):
    overlay = canvas.copy()
    for r in regions:
        overlay[r == 1] = color_bgr
    return cv2.addWeighted(canvas, 1.0 - alpha, overlay, alpha, 0)


def _iou_match(p_regions, g_regions, iou_thresh=0.5):
    """回傳已匹配的 pred index set 和 gt index set。"""
    if not p_regions or not g_regions:
        return set(), set()
    iou_mat = np.zeros((len(p_regions), len(g_regions)))
    for i, pm in enumerate(p_regions):
        for j, gm in enumerate(g_regions):
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou_mat[i, j] = inter / (union + 1e-8)
    matched_p, matched_g = set(), set()
    for i in range(len(p_regions)):
        best_g, best_iou = -1, -1.0
        for j in range(len(g_regions)):
            if j in matched_g:
                continue
            if iou_mat[i, j] > best_iou:
                best_iou, best_g = iou_mat[i, j], j
        if best_iou >= iou_thresh:
            matched_p.add(i)
            matched_g.add(best_g)
    return matched_p, matched_g


def save_ll_vis_single(img_bgr, p_bin_nat, g_bin_nat, metrics, save_path, lane_width_nat):
    """
    3 欄對比圖：GT | 預測 | Overlay（TP/FP/FN 色彩標注）

    顏色定義（BGR）：
      GT 車道線          → 藍色  (255, 0, 0)
      預測車道線         → 綠色  (0, 255, 0)
      TP GT（命中）      → 青色  (255, 255, 0)
      TP Pred（命中）    → 草綠  (0, 255, 128)
      FN（漏偵 GT）      → 紅色  (0, 0, 255)
      FP（誤判 Pred）    → 橙色  (0, 165, 255)
    """
    H, W = img_bgr.shape[:2]

    p_regions = get_pred_30px_regions(p_bin_nat, lane_width=lane_width_nat)
    g_regions = get_gt_30px_regions(g_bin_nat,   lane_width=lane_width_nat, min_pixels=15)
    matched_p, matched_g = _iou_match(p_regions, g_regions, iou_thresh=0.5)

    # Panel 1: GT（藍）
    vis_gt = _overlay_regions(img_bgr.copy(),
                              g_regions, (255, 0, 0))

    # Panel 2: Prediction（綠）
    vis_pred = _overlay_regions(img_bgr.copy(),
                                p_regions, (0, 255, 0))

    # Panel 3: Overlay
    vis_ov = img_bgr.copy()
    # FN（漏偵）
    vis_ov = _overlay_regions(vis_ov,
                              [g_regions[j] for j in range(len(g_regions)) if j not in matched_g],
                              (0, 0, 255))
    # FP（誤判）
    vis_ov = _overlay_regions(vis_ov,
                              [p_regions[i] for i in range(len(p_regions)) if i not in matched_p],
                              (0, 165, 255))
    # TP GT → 青
    vis_ov = _overlay_regions(vis_ov,
                              [g_regions[j] for j in matched_g],
                              (255, 255, 0))
    # TP Pred → 草綠
    vis_ov = _overlay_regions(vis_ov,
                              [p_regions[i] for i in matched_p],
                              (0, 255, 128))

    final = np.hstack([vis_gt, vis_pred, vis_ov])

    # 圖例文字
    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscl  = max(0.45, H / 800.0)
    thick = max(1, H // 400)
    texts = [
        (10,         "GT  (Blue)"),
        (W + 10,     f"Pred (Green)  F1@0.5={metrics['f1']:.3f}"
                     f"  P={metrics['precision']:.2f}  R={metrics['recall']:.2f}"),
        (W * 2 + 10, f"Overlay  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}"
                     f"  PixelIoU={metrics['pixel_iou']:.3f}"),
    ]
    y = max(25, int(H * 0.045))
    for x_off, text in texts:
        cv2.putText(final, text, (x_off, y), font, fscl, (0, 0, 0),   thick + 1, cv2.LINE_AA)
        cv2.putText(final, text, (x_off, y), font, fscl, (255, 255, 255), thick, cv2.LINE_AA)

    cv2.imwrite(save_path, final)


# =============================================================================
# Main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Find top-K LL images by per-image F1@0.5")
    p.add_argument("--checkpoint",  required=True,           help=".pth checkpoint 路徑")
    p.add_argument("--config",      required=True,           help="JSON config 路徑")
    p.add_argument("--img_dir",     required=True,           help="圖片資料夾路徑")
    p.add_argument("--ann_dir",     required=True,           help="標籤資料夾路徑（binary PNG masks）")
    p.add_argument("--top_k",       type=int, default=10,    help="輸出前幾名（預設 10）")
    p.add_argument("--output",      default="top_ll_images", help="輸出目錄（預設 top_ll_images）")
    p.add_argument("--prob_thresh", type=float, default=0.3, help="Sigmoid 閾值（預設 0.3）")
    p.add_argument("--no_vis",      action="store_true",     help="只輸出名稱清單，不存視覺化圖片")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. 載入設定 ──────────────────────────────────────────────────────
    print(f"Loading config: {args.config}")
    cfg = MultiTaskTrainingConfig.load(args.config)

    # ── 2. 建立 DataLoader ───────────────────────────────────────────────
    transforms_ll_val = [
        transform.LoadImg(),
        transform.ToTensor(),
        transform.Resize([360, 640]),   # 對齊訓練/評估解析度
        transform.Normalize(),
        Cleanup(),
    ]
    ds = get_dataset(cfg.dataset_ll, args.img_dir, args.ann_dir, None, transforms_ll_val)
    if ds is None or len(ds) == 0:
        print(f"Error: No data found at img_dir={args.img_dir}")
        return
    print(f"Dataset: {len(ds)} images  |  Device: {device}")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=cfg.pin_memory)

    # ── 3. 建立模型並載入 checkpoint ─────────────────────────────────────
    print("Building model and loading checkpoint...")
    config_rm = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_rm.num_labels = len(Category.load(cfg.category_csv_rlmd))
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
        ema_data   = ckpt['ema']
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

    missing, _ = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"  [Checkpoint] Missing keys: {len(missing)}")
    model.eval()

    # ── 4. Per-image F1 評估迴圈 ─────────────────────────────────────────
    print("\nComputing per-image F1@0.5...")
    records = []   # [(img_path, metrics, logit_np, gt_np, orig_w, orig_h)]

    autocast_enabled = (device.type == 'cuda')
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=autocast_enabled):
        for data in tqdm(dl, desc="Evaluating", dynamic_ncols=True):
            gt_mask = data.get("lane_mask")
            if gt_mask is None:
                continue

            img      = data["img"].to(device)
            gt_mask  = gt_mask.to(device)
            img_path = data["img_path"][0]

            orig_size = data.get("orig_size")
            if orig_size is not None:
                orig_w = int(orig_size[0][0]) if isinstance(orig_size[0], torch.Tensor) else int(orig_size[0])
                orig_h = int(orig_size[1][0]) if isinstance(orig_size[1], torch.Tensor) else int(orig_size[1])
            else:
                orig_img = cv2.imread(img_path)
                orig_h, orig_w = (orig_img.shape[:2] if orig_img is not None else (720, 1280))

            # Forward
            outputs    = model(pixel_values=img, task="ll")
            logits     = outputs.get("mask_logits", outputs.get("logits"))  # [1,1,H,W]

            # GT 對齊模型輸出解析度
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.unsqueeze(1)
            if gt_mask.shape[-2:] != logits.shape[-2:]:
                gt_mask = F.interpolate(gt_mask.float(), size=logits.shape[-2:], mode="nearest")

            logit_np = logits.squeeze().cpu().float().numpy()   # [H, W]
            gt_np    = gt_mask.squeeze().cpu().float().numpy()  # [H, W]

            metrics = compute_per_image_ll_f1(
                logit_np, gt_np,
                prob_thresh=args.prob_thresh,
                lane_width=30,
                iou_thresh=0.5,
            )
            records.append((img_path, metrics, logit_np, gt_np, orig_w, orig_h))

    # ── 5. 排序並取 Top-K ─────────────────────────────────────────────────
    records.sort(key=lambda x: x[1]["f1"], reverse=True)
    top_records = records[:args.top_k]

    # ── 6. 輸出排名表 ─────────────────────────────────────────────────────
    header = f"TOP {args.top_k} IMAGES (LL F1@0.5)"
    print(f"\n{header:^72}")
    print("=" * 72)
    print(f"{'Rank':<6} {'F1@0.5':>8} {'Prec':>7} {'Recall':>7} {'PixIoU':>8}  Filename")
    print("-" * 72)
    for i, (img_path, m, *_) in enumerate(top_records):
        fname = os.path.basename(img_path)
        print(f"{i+1:<6} {m['f1']:>8.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['pixel_iou']:>8.4f}  {fname}")
    print("=" * 72)

    os.makedirs(args.output, exist_ok=True)
    txt_path = os.path.join(args.output, "top_images.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Top {args.top_k} LL images by F1@0.5\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"img_dir:    {args.img_dir}\n")
        f.write("=" * 72 + "\n")
        f.write(f"{'Rank':<6} {'F1@0.5':>8} {'Prec':>7} {'Recall':>7} {'PixIoU':>8}  Filename\n")
        f.write("-" * 72 + "\n")
        for i, (img_path, m, *_) in enumerate(top_records):
            fname = os.path.basename(img_path)
            f.write(f"{i+1:<6} {m['f1']:>8.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['pixel_iou']:>8.4f}  {fname}\n")
        f.write("=" * 72 + "\n")
    print(f"\nFilename list saved: {txt_path}")

    # ── 7. 視覺化 ────────────────────────────────────────────────────────
    if not args.no_vis:
        print(f"\nSaving visualization images to: {args.output}/")
        for i, (img_path, metrics, logit_np, gt_np, orig_w, orig_h) in enumerate(top_records):
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  [Skip] Cannot read: {img_path}")
                continue

            # 視覺化時放大回原圖解析度，呈現效果較佳
            p_prob = 1.0 / (1.0 + np.exp(-logit_np.astype(np.float64)))
            p_bin_nat = cv2.resize(
                (p_prob > args.prob_thresh).astype(np.uint8),
                (orig_w, orig_h), interpolation=cv2.INTER_NEAREST,
            )
            g_bin_nat = cv2.resize(
                (gt_np > 0.5).astype(np.uint8),
                (orig_w, orig_h), interpolation=cv2.INTER_NEAREST,
            )

            # lane_width 等比例換算到原圖解析度
            lane_width_nat = max(20, int(30 * orig_h / 360))

            fname_stem = os.path.splitext(os.path.basename(img_path))[0]
            save_path  = os.path.join(
                args.output,
                f"rank{i+1:02d}_f1{metrics['f1']:.4f}_{fname_stem}.jpg"
            )
            save_ll_vis_single(img_bgr, p_bin_nat, g_bin_nat, metrics, save_path, lane_width_nat)
            print(f"  [{i+1:02d}] F1={metrics['f1']:.4f}  {os.path.basename(save_path)}")

    print("\nDone!")


if __name__ == "__main__":
    main()
