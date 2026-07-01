"""
inference_groundtruth.py
────────────────────────────────────────────────────────────────────────
多任務 Ground Truth 視覺化程式。
輸出格式與 inference_multitask.py 完全對齊，方便直接比對。

支援任務：
  rm    — 道路標線語意分割 GT（調色盤疊加，同 inference RM 輸出）
  ll    — 車道線 GT 實例折線（與 inference LL 相同 pipeline，套用於 GT mask）
  llseg — 車道線 GT 純語意分割遮罩（綠色半透明，同 inference LLSEG 輸出）
  ts    — 交通標誌 GT BBox（.txt 檔，格式：cls_name x1 y1 x2 y2）
  tl    — 交通號誌 GT BBox（.txt 檔，格式：cls_name x1 y1 x2 y2）

使用範例：
  python tools/inference_groundtruth.py \\
    --config configs/train_uda_multi_tasks.json \\
    --input  /path/to/images \\
    --ann    /path/to/annotations \\
    --output gt_results \\
    --task   ll
"""

import argparse
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from configs.multitask_config import MultiTaskTrainingConfig
from engine.category import Category


# =============================================================================
# LL helper functions（完全對齊 inference_multitask.py）
# =============================================================================

def _lane_fit_pca(xs, ys, orig_h, orig_w):
    pts_all = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    mean_pt = pts_all.mean(axis=0)
    centered = pts_all - mean_pt

    cov = np.cov(centered.T)
    _, evecs = np.linalg.eigh(cov)
    primary = evecs[:, 1]
    if primary[1] < 0:
        primary = -primary

    proj = centered @ primary
    proj_range = float(proj.max() - proj.min())
    if proj_range < 20:
        return None

    n_bins = int(np.clip(proj_range / 8, 15, 80))
    edges = np.linspace(proj.min(), proj.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(proj, edges) - 1, 0, n_bins - 1)

    t_list, cx_list, cy_list = [], [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() < 3:
            continue
        cx, cy = (centered[m].mean(axis=0) + mean_pt).tolist()
        t_list.append(float(proj[m].mean()))
        cx_list.append(cx)
        cy_list.append(cy)

    if len(t_list) < 5:
        return None

    t_arr  = np.array(t_list)
    cx_arr = np.array(cx_list)
    cy_arr = np.array(cy_list)

    n  = len(t_arr)
    lo = max(0, int(n * 0.05))
    hi = min(n, int(n * 0.95) + 1)
    t_fit, cx_fit, cy_fit = t_arr[lo:hi], cx_arr[lo:hi], cy_arr[lo:hi]
    if len(t_fit) < 5:
        return None

    deg = 2 if proj_range > orig_h * 0.25 else 1
    try:
        px = np.polyfit(t_fit, cx_fit, deg=deg)
        py = np.polyfit(t_fit, cy_fit, deg=deg)
    except np.linalg.LinAlgError:
        return None

    t_dense = np.linspace(t_fit.min(), t_fit.max(), 200)
    x_dense = np.polyval(px, t_dense)
    y_dense = np.polyval(py, t_dense)

    valid = ((x_dense >= 0) & (x_dense < orig_w) &
             (y_dense >= 0) & (y_dense < orig_h))
    if not valid.any():
        return None

    out = np.stack([x_dense[valid], y_dense[valid]], axis=1).astype(np.int32)
    return out if len(out) >= 2 else None


def _maybe_split_component(xs, ys, orig_w):
    x_span = float(xs.max() - xs.min())
    if x_span <= orig_w * 0.10:
        return [(xs, ys)]

    n_bins = max(10, int(x_span / 5))
    x_hist, x_edges = np.histogram(xs, bins=n_bins)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2

    n = len(x_hist)
    lo, hi = int(n * 0.2), int(n * 0.8)
    if lo >= hi:
        return [(xs, ys)]

    valley_idx = lo + int(np.argmin(x_hist[lo:hi]))
    if x_hist[valley_idx] > x_hist.max() * 0.30:
        return [(xs, ys)]

    split_x = float(x_centers[valley_idx])
    l_mask = xs < split_x
    r_mask = ~l_mask

    groups = []
    if l_mask.sum() >= 20:
        groups.append((xs[l_mask], ys[l_mask]))
    if r_mask.sum() >= 20:
        groups.append((xs[r_mask], ys[r_mask]))
    return groups if len(groups) == 2 else [(xs, ys)]


def _group_nearby_components(components, orig_h, orig_w):
    if not components:
        return []

    n         = len(components)
    x_thresh  = orig_w * 0.05
    y_gap_max = orig_h * 0.15

    def get_info(xs, ys):
        y_range = float(ys.max() - ys.min())
        q_lo = ys.min() + y_range * 0.2
        q_hi = ys.max() - y_range * 0.2
        top_mask = ys <= q_lo
        bot_mask = ys >= q_hi
        top_x = float(np.median(xs[top_mask])) if top_mask.any() else float(xs[int(np.argmin(ys))])
        bot_x = float(np.median(xs[bot_mask])) if bot_mask.any() else float(xs[int(np.argmax(ys))])
        top_y = float(ys[top_mask].mean()) if top_mask.any() else float(ys.min())
        bot_y = float(ys[bot_mask].mean()) if bot_mask.any() else float(ys.max())
        dy    = bot_y - top_y
        slope = (bot_x - top_x) / dy if abs(dy) > 1 else 0.0
        return top_y, top_x, bot_y, bot_x, slope

    info   = [get_info(xs, ys) for xs, ys in components]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            ty_i, tx_i, by_i, bx_i, sl_i = info[i]
            ty_j, tx_j, by_j, bx_j, sl_j = info[j]
            if ty_i > ty_j:
                ty_i, tx_i, by_i, bx_i, sl_i, \
                ty_j, tx_j, by_j, bx_j, sl_j = \
                ty_j, tx_j, by_j, bx_j, sl_j, \
                ty_i, tx_i, by_i, bx_i, sl_i
            if by_i >= ty_j:
                continue
            y_gap = ty_j - by_i
            if y_gap > y_gap_max:
                continue
            pred_x_from_i = bx_i + sl_i * y_gap
            pred_x_from_j = tx_j - sl_j * y_gap
            if abs(pred_x_from_i - tx_j) < x_thresh or \
               abs(pred_x_from_j - bx_i) < x_thresh:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idxs in groups.values():
        all_xs = np.concatenate([components[k][0] for k in idxs])
        all_ys = np.concatenate([components[k][1] for k in idxs])
        merged.append((all_xs, all_ys))
    return merged


def _extract_ll_instance_curves(bin_mask):
    """
    GT binary mask → 實例折線點陣列表（與 inference_multitask.py LL pipeline 完全一致）
    """
    orig_h, orig_w = bin_mask.shape

    bin_mask_orig = bin_mask.copy()
    kernel_v     = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    bin_mask     = cv2.dilate(bin_mask, kernel_v, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    bin_mask     = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    min_pixels      = max(80, orig_h)
    min_orig_height = max(20, int(orig_h * 0.04))

    raw_components = []
    for lab in range(1, n_labels):
        if stats[lab, cv2.CC_STAT_AREA] < min_pixels:
            continue
        ys, xs = np.where(labels == lab)
        if len(ys) < 20:
            continue
        orig_ys = np.where((labels == lab) & (bin_mask_orig == 1))[0]
        if len(orig_ys) == 0:
            continue
        if orig_ys.max() - orig_ys.min() < min_orig_height:
            continue
        for sub_xs, sub_ys in _maybe_split_component(xs, ys, orig_w):
            if len(sub_ys) >= 20:
                raw_components.append((sub_xs, sub_ys))

    grouped = _group_nearby_components(raw_components, orig_h, orig_w)

    lane_curves = []
    min_curve_len = max(50, orig_h * 0.05)
    for g_xs, g_ys in grouped:
        pts = _lane_fit_pca(g_xs, g_ys, orig_h, orig_w)
        if pts is None:
            continue
        diffs   = np.diff(pts.astype(float), axis=0)
        arc_len = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
        if arc_len < min_curve_len:
            continue
        lane_curves.append((float(np.median(g_xs)), pts))

    lane_curves.sort(key=lambda t: t[0])
    return [pts for _, pts in lane_curves]


# =============================================================================
# 工具函式
# =============================================================================

def get_palette(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    categories = Category.load(csv_path)
    return np.array([[c.r, c.g, c.b] for c in categories], dtype=np.uint8)


def get_image_list(input_path):
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    if os.path.isdir(input_path):
        return sorted([
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])
    return [input_path]


def find_ann(ann_dir, stem, ext):
    """在 ann_dir 裡找 stem+ext，找不到回傳 None。"""
    p = os.path.join(ann_dir, stem + ext)
    return p if os.path.exists(p) else None


def _resolve_label_positions(items, H, W, font_scale=0.4):
    """與 evaluate_all_tasks.py 相同的貪婪標籤位置解算（避免文字重疊）。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad  = 3
    placed  = []
    results = []

    def overlaps_any(r):
        for p in placed:
            if r[0] < p[2] and r[2] > p[0] and r[1] < p[3] and r[3] > p[1]:
                return True
        return False

    for (x1, y1, x2, y2, text, color) in items:
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
        lw, lh = tw + pad * 2, th + pad * 2

        def clamp_rect(bx1, by1):
            bx1 = int(max(0, min(bx1, W - lw)))
            by1 = int(max(0, min(by1, H - lh)))
            return (bx1, by1, bx1 + lw, by1 + lh)

        candidates = [
            clamp_rect(x1,        y1 - lh),
            clamp_rect(x2 - lw,   y1 - lh),
            clamp_rect(x1,        y2),
            clamp_rect(x2,        y1),
            clamp_rect(x1 - lw,   y1),
            clamp_rect(x1,        y1 - lh * 2),
            clamp_rect(x2,        y2 - lh),
        ]
        chosen = next((r for r in candidates if not overlaps_any(r)), candidates[0])
        placed.append(chosen)
        bx1, by1, bx2, by2 = chosen
        results.append((bx1, by1, bx2, by2, bx1 + pad, by2 - pad, text, color))

    return results


# =============================================================================
# 各任務 GT 處理函式
# =============================================================================

def visualize_rm_gt(img_path, ann_dir, palette, opacity):
    """RM：讀取 PNG class-index 標籤 → 調色盤疊加。"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = find_ann(ann_dir, stem, '.png')

    original_img = cv2.imread(img_path)
    if original_img is None:
        return None
    vis = original_img.copy()

    if ann_path is None or palette is None:
        return vis

    # 必須用 PIL 讀取：RLMD 標籤是 P mode（調色盤索引）PNG，
    # cv2.IMREAD_GRAYSCALE 會把調色盤顏色轉成灰階值而非保留 class index。
    try:
        ann = np.asarray(Image.open(ann_path)).copy()
    except Exception:
        return vis

    orig_h, orig_w = original_img.shape[:2]
    if ann.shape != (orig_h, orig_w):
        ann = cv2.resize(ann, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    ann = np.clip(ann, 0, len(palette) - 1)   # 防呆：裁剪至合法範圍
    color_mask     = palette[ann]                                    # HWC, RGB
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    vis = cv2.addWeighted(original_img, 1 - opacity, color_mask_bgr, opacity, 0)
    return vis


def visualize_llseg_gt(img_path, ann_dir, ll_thresh, opacity):
    """LLSEG：讀取 binary PNG mask → 綠色半透明疊加（同 inference llseg 輸出）。"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = find_ann(ann_dir, stem, '.png')

    original_img = cv2.imread(img_path)
    if original_img is None:
        return None
    vis = original_img.copy()

    if ann_path is None:
        return vis

    ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
    if ann is None:
        return vis

    orig_h, orig_w = original_img.shape[:2]
    if ann.shape != (orig_h, orig_w):
        ann = cv2.resize(ann, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    bin_mask = (ann > int(ll_thresh * 255)).astype(np.uint8)

    color_mask = np.zeros_like(original_img)
    color_mask[bin_mask == 1] = [0, 255, 0]

    lane_region = bin_mask == 1
    if lane_region.any():
        alpha = 0.5
        vis[lane_region] = cv2.addWeighted(
            original_img[lane_region], 1 - alpha,
            color_mask[lane_region],   alpha, 0
        )

    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def visualize_ll_gt(img_path, ann_dir, ll_thresh):
    """LL：讀取 binary PNG mask → 跑實例 pipeline → 多彩折線（同 inference ll 輸出）。"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = find_ann(ann_dir, stem, '.png')

    original_img = cv2.imread(img_path)
    if original_img is None:
        return None
    vis = original_img.copy()

    if ann_path is None:
        return vis

    ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
    if ann is None:
        return vis

    orig_h, orig_w = original_img.shape[:2]
    if ann.shape != (orig_h, orig_w):
        ann = cv2.resize(ann, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    bin_mask   = (ann > int(ll_thresh * 255)).astype(np.uint8)
    lane_curves = _extract_ll_instance_curves(bin_mask)

    COLORS = [
        (  0, 255,   0),
        (  0, 200, 255),
        (255, 100,   0),
        (255,   0, 255),
        (  0, 100, 255),
        (100, 255, 100),
    ]
    LINE_WIDTH = max(3, orig_h // 170)

    overlay = vis.copy()
    for i, pts in enumerate(lane_curves):
        color = COLORS[i % len(COLORS)]
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)],
                      isClosed=False, color=color,
                      thickness=LINE_WIDTH, lineType=cv2.LINE_AA)

    vis = cv2.addWeighted(vis, 0.4, overlay, 0.6, 0)
    return vis


TL_SKIP_CLASSES = {"wait_on"}   # TL 任務中不繪製的類別

def visualize_det_gt(img_path, ann_dir, palette, categories, task="ts"):
    """TS / TL：讀取 YOLO-style .txt（cls_name x1 y1 x2 y2，絕對像素座標）→ BBox 疊加。"""
    stem = os.path.splitext(os.path.basename(img_path))[0]
    ann_path = find_ann(ann_dir, stem, '.txt')

    original_img = cv2.imread(img_path)
    if original_img is None:
        return None
    vis = original_img.copy()

    if ann_path is None:
        return vis

    orig_h, orig_w = original_img.shape[:2]

    # 建立 name → id 映射（由 Category CSV 載入的順序決定）
    name_to_id = {c.name: c.id for c in categories} if categories else {}

    items = []
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_name = parts[0]
            try:
                coords = [float(v) for v in parts[1:5]]
            except ValueError:
                continue

            x1, y1, x2, y2 = (int(c) for c in coords)
            x1 = max(0, x1);  y1 = max(0, y1)
            x2 = min(orig_w, x2);  y2 = min(orig_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            if task == 'tl' and cls_name in TL_SKIP_CLASSES:
                continue

            cls_id = name_to_id.get(cls_name, -1)
            if palette is not None and 0 <= cls_id < len(palette):
                c     = palette[cls_id]
                color = (int(c[2]), int(c[1]), int(c[0]))   # RGB → BGR
            else:
                color = (0, 200, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            items.append((x1, y1, x2, y2, cls_name, color))

    for bx1, by1, bx2, by2, tx, ty, text, col in _resolve_label_positions(
            items, orig_h, orig_w):
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), col, -1)
        cv2.putText(vis, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


# =============================================================================
# 主程式
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Task Ground Truth Visualizer")
    parser.add_argument("--config",   type=str, required=True,
                        help="Path to config json")
    parser.add_argument("--input",    type=str, required=True,
                        help="Path to image or directory")
    parser.add_argument("--ann",      type=str, required=True,
                        help="Path to annotation directory")
    parser.add_argument("--output",   type=str, default="gt_results",
                        help="Output directory")
    parser.add_argument("--task",     type=str, required=True,
                        choices=["rm", "ll", "llseg", "ts", "tl"],
                        help="Task name")
    parser.add_argument("--opacity",  type=float, default=0.6,
                        help="Mask opacity for RM task")
    parser.add_argument("--ll_thresh", type=float, default=0.5,
                        help="Binarization threshold for LL GT mask (0–1, default 0.5)")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg        = MultiTaskTrainingConfig.load(args.config)
    image_list = get_image_list(args.input)
    if not image_list:
        print("No images found!")
        return

    save_dir = os.path.join(args.output, args.task)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[GT] task={args.task} | {len(image_list)} images → {save_dir}")

    # ── 載入調色盤 / 類別資訊 ────────────────────────────────────────
    palette    = None
    categories = []

    if args.task == 'rm':
        palette = get_palette(cfg.category_csv_rlmd)

    elif args.task in ['ll', 'llseg']:
        pass   # 不需要調色盤

    elif args.task == 'ts':
        csv = getattr(cfg, 'category_csv_ts', None)
        if csv and os.path.exists(csv):
            categories = Category.load(csv)
            palette    = get_palette(csv)
        else:
            print("[Warning] TS CSV not found, boxes will all be drawn in green.")

    elif args.task == 'tl':
        csv = getattr(cfg, 'category_csv_tl', None)
        if csv and os.path.exists(csv):
            categories = Category.load(csv)
            palette    = get_palette(csv)
        else:
            print("[Warning] TL CSV not found, boxes will all be drawn in green.")

    # ── 推論迴圈 ─────────────────────────────────────────────────────
    for img_path in tqdm(image_list):
        stem = os.path.splitext(os.path.basename(img_path))[0]

        if args.task == 'rm':
            result = visualize_rm_gt(img_path, args.ann, palette, args.opacity)

        elif args.task == 'llseg':
            result = visualize_llseg_gt(img_path, args.ann, args.ll_thresh, args.opacity)

        elif args.task == 'll':
            result = visualize_ll_gt(img_path, args.ann, args.ll_thresh)

        elif args.task in ['ts', 'tl']:
            result = visualize_det_gt(img_path, args.ann, palette, categories, task=args.task)

        else:
            result = None

        if result is None:
            print(f"[Skip] {img_path}")
            continue

        cv2.imwrite(os.path.join(save_dir, f"{stem}.png"), result)

    print("\nDone!")


if __name__ == "__main__":
    main()
