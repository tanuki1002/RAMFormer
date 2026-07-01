import argparse
import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as T
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from configs.multitask_config import MultiTaskTrainingConfig
from engine.multi_task_segformer import get_model as get_mt_model
from engine.category import Category
from engine.decode_utils import decode_yolox_outputs
from transformers import SegformerConfig

def slide_inference_rm(model, tensor, num_classes, crop_size, stride, device):
    """Sliding window inference — matches Validator.slide_inference."""
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    _, _, h_img, w_img = tensor.shape

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    preds     = tensor.new_zeros((1, num_classes, h_img, w_img))
    count_mat = tensor.new_zeros((1, 1, h_img, w_img))
    autocast_enabled = (device.type == 'cuda')

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            crop = tensor[:, :, y1:y2, x1:x2]
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=autocast_enabled):
                out = model(pixel_values=crop, task='rm')
            crop_logits = out['logits'] if isinstance(out, dict) else out[0]

            preds += F.pad(
                crop_logits,
                (int(x1), int(preds.shape[3] - x2),
                 int(y1), int(preds.shape[2] - y2)),
            )
            count_mat[:, :, y1:y2, x1:x2] += 1

    return preds / count_mat


def rm_post_process(pred_mask: np.ndarray, min_area: int = 400) -> np.ndarray:
    """
    RM 後處理：
      1. 移除面積 < min_area 的碎片（→ 背景 0）
      2. 對每個 connected component 做 majority vote，修正類別分裂
    """
    result = pred_mask.copy()
    fg = (pred_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    for lab in range(1, num_labels):
        area = stats[lab, cv2.CC_STAT_AREA]
        mask = labels == lab

        if area < min_area:
            result[mask] = 0          # 小碎片 → 背景
            continue

        # majority vote：取該 component 內最多的非零 class
        classes = pred_mask[mask]
        fg_ids  = classes[classes > 0]
        if len(fg_ids) == 0:
            continue
        vals, cnts = np.unique(fg_ids, return_counts=True)
        result[mask] = vals[cnts.argmax()]

    return result




def _lane_fit_pca(xs, ys, orig_h, orig_w):
    """
    PCA 主軸方向擬合：無論車道線方向（垂直、斜向），均沿像素雲的主軸擬合。

    步驟：
      1. PCA 求主軸向量（任意方向）
      2. 沿主軸投影，分 bin 取 2D centroid
      3. 對 centroid 序列做參數化多項式擬合 x(t), y(t)
         - 短跨度用 deg=1（線性），避免曲率 artifact
         - 長跨度用 deg=2（二次）
      4. 前後各縮 5% bin 避免端點外推發散
      5. 密集求值，輸出平滑曲線點陣

    相較於 polyfit(ys, xs)：
      - 不假設線段方向接近垂直
      - 對斜向/橫向車道線同樣有效
    """
    pts_all = np.stack([xs.astype(float), ys.astype(float)], axis=1)
    mean_pt = pts_all.mean(axis=0)
    centered = pts_all - mean_pt

    cov = np.cov(centered.T)
    _, evecs = np.linalg.eigh(cov)
    primary = evecs[:, 1]          # 最大特徵值方向 = 主軸
    if primary[1] < 0:             # 讓主軸指向 y 增大（靠近鏡頭方向）
        primary = -primary

    proj = centered @ primary
    proj_range = float(proj.max() - proj.min())
    if proj_range < 20:
        return None

    # 沿主軸分 bin，每 bin 計算 2D centroid
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

    # 前後各縮 5% bin，避免端點外推發散
    n  = len(t_arr)
    lo = max(0, int(n * 0.05))
    hi = min(n, int(n * 0.95) + 1)
    t_fit, cx_fit, cy_fit = t_arr[lo:hi], cx_arr[lo:hi], cy_arr[lo:hi]
    if len(t_fit) < 5:
        return None

    # 短投影跨度用 deg=1，長跨度用 deg=2
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


def _group_nearby_components(components, orig_h, orig_w):
    """
    將同一條虛線車道的相鄰 component 合併成一個像素集，再統一擬合。

    合併條件（同時滿足）：
      1. Y range 不重疊（一個在另一個上方）
      2. Y 間距 < 15% 圖高
      3. 方向延伸吻合：將上方 component 的斜率延伸到下方 component 的起始 Y，
         預測的 X 與實際 X 之差 < 5% 圖寬（反方向同理）

    重點：用方向延伸而非單純比較端點 X，避免將不同方向但位置接近的線段錯誤合併。
    """
    if not components:
        return []

    n         = len(components)
    x_thresh  = orig_w * 0.05   # 5% 圖寬
    y_gap_max = orig_h * 0.15   # 15% 圖高

    def get_info(xs, ys):
        """
        用各 component 頂/底 20% 像素的中位數 X 估計斜率，比用極端點更穩健。
        回傳 (top_y, top_x, bot_y, bot_x, slope)
        """
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

    info = [get_info(xs, ys) for xs, ys in components]

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
            # 確保 i 在上（較小 y），j 在下
            if ty_i > ty_j:
                ty_i, tx_i, by_i, bx_i, sl_i, \
                ty_j, tx_j, by_j, bx_j, sl_j = \
                ty_j, tx_j, by_j, bx_j, sl_j, \
                ty_i, tx_i, by_i, bx_i, sl_i
            # Y range 不能重疊
            if by_i >= ty_j:
                continue
            y_gap = ty_j - by_i
            if y_gap > y_gap_max:
                continue
            # 用 i 的斜率預測在 j 起點的 X
            pred_x_from_i = bx_i + sl_i * y_gap
            # 用 j 的斜率反向預測在 i 終點的 X
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


def _maybe_split_component(xs, ys, orig_w):
    """
    若一個 connected component 在 X 方向跨度過大（可能是兩條相鄰車道線在 mask 上相連），
    嘗試透過 X 直方圖的谷點將其拆分成兩個子群組，各自獨立擬合。

    判斷邏輯：
    - X 跨度 > 10% 圖寬 → 可能是兩條線合併
    - 在 X 直方圖中間 60% 找谷點，若谷值 < 30% 最大值 → 有明顯分界線
    - 否則不拆分，視為單條線
    """
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
        return [(xs, ys)]  # 沒有明顯谷點，不拆分

    split_x = float(x_centers[valley_idx])
    l_mask = xs < split_x
    r_mask = ~l_mask

    groups = []
    if l_mask.sum() >= 20:
        groups.append((xs[l_mask], ys[l_mask]))
    if r_mask.sum() >= 20:
        groups.append((xs[r_mask], ys[r_mask]))
    return groups if len(groups) == 2 else [(xs, ys)]


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Multi-Task Segformer")
    parser.add_argument("--config",     type=str, required=True,  help="Path to config json")
    parser.add_argument("--checkpoint", type=str, required=True,  help="Path to checkpoint (.pth)")
    parser.add_argument("--input",      type=str, required=True,  help="Path to image or directory")
    parser.add_argument("--output",     type=str, default="inference_results", help="Output directory")
    parser.add_argument("--task",       type=str, default="rm",
                        choices=["rm", "ll", "ts", "tl", "llseg"], help="Task name")
    parser.add_argument("--opacity",    type=float, default=0.6,  help="Mask opacity (rm task)")
    parser.add_argument("--conf_thresh",   type=float, default=0.5,  help="Confidence threshold for TS")
    parser.add_argument("--tl_conf_thresh",type=float, default=0.3,  help="Confidence threshold for TL")
    parser.add_argument("--ll_thresh",  type=float, default=0.4,
                        help="Existence probability threshold for LL lanes (0.1~0.9)")
    return parser.parse_args()


def get_palette(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    categories = Category.load(csv_path)
    return np.array([[c.r, c.g, c.b] for c in categories], dtype=np.uint8)


def preprocess_image(img_path, target_h, target_w):
    """
    讀取圖片，resize 到 (target_h, target_w)（32 的倍數），回傳 tensor 與原始尺寸。
    """
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Cannot read {img_path}")
        return None, None, None

    original_size = image.shape[:2]          # (H, W)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = TF.to_tensor(image)             # [0, 1], shape [3, H, W]
    tensor = TF.normalize(tensor,
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    # 確保尺寸是 32 的倍數（SegFormer 必要條件）
    new_h = int(round(target_h / 32) * 32)
    new_w = int(round(target_w / 32) * 32)
    tensor = TF.resize(tensor, (new_h, new_w),
                       interpolation=InterpolationMode.BILINEAR, antialias=True)

    return tensor.unsqueeze(0), original_size, (new_h, new_w)


def preprocess_rm_views(img_path, target_h, target_w, cfg):
    """
    RM-specific preprocessing: applies the first ContrastStretch (log) to match training.
    Only the log view is used at inference because the exp view values are near-zero
    (~0.001–0.008 after the per-image normalisation), which degrades predictions when
    averaged.  Returns a list of one [1, 3, H, W] tensor, original_size, resized_size.
    """
    from engine.transform import ContrastStretch as CS

    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Error: Cannot read {img_path}")
        return None, None, None

    original_size = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Apply only the first stretch (log) — closest to the training distribution
    # while keeping visual discriminability high.
    fname  = cfg.contrast_stretch[0]
    param  = cfg.img_proc_params[0]
    data   = {"img": image_rgb}
    cs     = CS(max_intensity=cfg.max_intensity, function_name=fname, parameter=param)
    data   = cs.transform(data)
    # data["imgs"][0]: float32 numpy [H, W, 3], values in [0, 1]

    new_h = int(round(target_h / 32) * 32)
    new_w = int(round(target_w / 32) * 32)

    view = data["imgs"][0]
    t = TF.to_tensor(view)
    t = TF.resize(t, (new_h, new_w),
                  interpolation=InterpolationMode.BILINEAR, antialias=True)
    t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return [t.unsqueeze(0)], original_size, (new_h, new_w)


# decode_yolox_outputs is imported from engine.decode_utils

# =================================================================================
# 主程式
# =================================================================================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. 載入設定 ──────────────────────────────────────────────────
    cfg = MultiTaskTrainingConfig.load(args.config)
    print(f"Loaded config: {args.config}")

    # ── 2. 決定推論尺寸 ──────────────────────────────────────────────
    if args.task in ['ts', 'tl']:
        inference_scale = (960, 960)
        print(f"[Scale] Override to (960, 960) for {args.task.upper()}")
    elif args.task in ['ll', 'llseg']:
        # 車道線必須與訓練 crop_size 保持一致，使 row_ys 對齊
        inference_scale = tuple(cfg.crop_size)   # 通常 (512, 512)
        print(f"[Scale] Override to {inference_scale} for LL (matches training crop_size)")
    else:
        inference_scale = tuple(cfg.image_scale)

    # 訓練時的模型輸入空間（用於 LL 座標還原）
    model_h, model_w = cfg.crop_size[0], cfg.crop_size[1]

    # ── 3. 顏色盤 ────────────────────────────────────────────────────
    palette  = None
    cats_ts  = []
    cats_tl  = []

    if args.task == 'rm':
        palette = get_palette(cfg.category_csv_rlmd)
    elif args.task in ['ll', 'llseg']:
        palette = get_palette(cfg.category_csv_ll)   # 僅備用，LL 不用 palette 畫線
    elif args.task == 'ts':
        if getattr(cfg, 'category_csv_ts', None) and os.path.exists(cfg.category_csv_ts):
            cats_ts = Category.load(cfg.category_csv_ts)
            palette = get_palette(cfg.category_csv_ts)
        else:
            print("Warning: TS CSV not found.")
    elif args.task == 'tl':
        if getattr(cfg, 'category_csv_tl', None) and os.path.exists(cfg.category_csv_tl):
            cats_tl = Category.load(cfg.category_csv_tl)
            palette = get_palette(cfg.category_csv_tl)
        else:
            print("Warning: TL CSV not found.")

    # ── 4. 建立模型 ──────────────────────────────────────────────────
    print("Building model...")
    segconfig_rm = SegformerConfig.from_pretrained("nvidia/mit-b1")
    segconfig_rm.num_labels = len(Category.load(cfg.category_csv_rlmd))

    segconfig_ll = SegformerConfig.from_pretrained("nvidia/mit-b1")
    segconfig_ll.num_labels = len(Category.load(cfg.category_csv_ll))

    cats_ts_for_config = cats_ts if cats_ts else []
    segconfig_ts = SegformerConfig.from_pretrained("nvidia/mit-b1")
    segconfig_ts.num_labels = (len(cats_ts_for_config)
                                if len(cats_ts_for_config) > 0
                                else getattr(cfg, 'ts_num_classes', 13))

    config_tl = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_tl.num_labels = getattr(cfg, 'tl_num_classes', 4)

    model = get_mt_model(segconfig_rm, segconfig_ll, segconfig_ts, config_tl)

    # ── 5. 載入權重 ──────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # 過濾掉 discriminator 權重（推論不需要）
    clean_sd = {k.replace("module.", ""): v
                for k, v in state_dict.items()
                if "discriminator" not in k}

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.to(device)
    model.eval()

    # ── 6. 準備圖片列表 ──────────────────────────────────────────────
    if os.path.isdir(args.input):
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_list = sorted([
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if os.path.splitext(f)[1].lower() in valid_ext
        ])
    else:
        image_list = [args.input]

    if not image_list:
        print("No images found!")
        return

    save_dir = os.path.join(args.output, args.task)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output dir: {save_dir}")
    print(f"Inference on {len(image_list)} images (task={args.task}) ...")

    # ── 7. 推論迴圈 ──────────────────────────────────────────────────
    for img_path in tqdm(image_list):
        img_stem = os.path.splitext(os.path.basename(img_path))[0]

        target_h, target_w = inference_scale
        if args.task == 'rm' and getattr(cfg, 'contrast_stretch', None):
            rm_views, original_size, resized_size = preprocess_rm_views(
                img_path, target_h, target_w, cfg)
            if rm_views is None:
                continue
            rm_views = [t.to(device) for t in rm_views]
            input_tensor = rm_views[0]
        else:
            rm_views = None
            input_tensor, original_size, resized_size = preprocess_image(
                img_path, target_h, target_w)
            if input_tensor is None:
                continue
            input_tensor = input_tensor.to(device)

        orig_h, orig_w = original_size
        res_h,  res_w  = resized_size          # 實際餵給模型的尺寸（32 的倍數）
        scale_x = orig_w / res_w
        scale_y = orig_h / res_h

        autocast_enabled = (device.type == 'cuda')
        # RM uses slide_inference (see below); other tasks use single-pass
        if args.task != 'rm':
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=autocast_enabled):
                model_task = 'll' if args.task == 'llseg' else args.task
                outputs = model(pixel_values=input_tensor, task=model_task)
        else:
            outputs = None

        # ────────────────────────────────────────────────────────────
        # 任務 A：TS / TL（物件偵測）
        # ────────────────────────────────────────────────────────────
        if args.task in ['ts', 'tl']:
            multi_scale_outputs = outputs['logits']
            _conf = args.tl_conf_thresh if args.task == 'tl' else args.conf_thresh
            final_bboxes, final_scores, final_classes = decode_yolox_outputs(
                multi_scale_outputs,
                conf_thresh=_conf,
                nms_thresh=0.45
            )

            bboxes = final_bboxes[0].cpu().numpy()  if (final_bboxes and final_bboxes[0].numel() > 0) else []
            scores = final_scores[0].cpu().numpy()  if (final_scores  and final_scores[0].numel()  > 0) else []
            clses  = final_classes[0].cpu().numpy() if (final_classes and final_classes[0].numel() > 0) else []

            vis_img = cv2.imread(img_path)
            active_cats = cats_ts if args.task == 'ts' else cats_tl
            # ── 兩遍式標籤放置：先畫框，再找不重疊位置放標籤 ──────────

            def _overlaps(a, b):
                return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

            # 第一遍：過濾 + 畫 bounding box + 收集有效偵測
            active_cats = cats_ts if args.task == 'ts' else cats_tl
            valid_dets  = []   # (x1,y1,x2,y2, cls_id, color, label_text)
            all_bboxes  = []   # 所有有效框的 rect

            for i in range(len(scores)):
                x1 = int(bboxes[i][0] * scale_x); y1 = int(bboxes[i][1] * scale_y)
                x2 = int(bboxes[i][2] * scale_x); y2 = int(bboxes[i][3] * scale_y)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(orig_w, x2); y2 = min(orig_h, y2)
                cls_id = int(clses[i])

                # ── TL 車燈誤判過濾 ─────────────────────────────────
                if args.task == 'tl':
                    box_w      = x2 - x1
                    box_h_size = max(y2 - y1, 1)
                    box_cy     = (y1 + y2) / 2
                    if box_cy > orig_h * 0.65 and box_w > orig_w * 0.10:
                        continue
                    if box_w / box_h_size > 2.5:
                        continue
                    if cls_id == 2 and scores[i] < max(args.tl_conf_thresh, 0.55):
                        continue
                # ────────────────────────────────────────────────────

                if palette is not None and cls_id < len(palette):
                    c = palette[cls_id]
                    color = (int(c[2]), int(c[1]), int(c[0]))
                else:
                    color = (0, 255, 0)

                cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                cls_name   = (active_cats[cls_id].name
                              if (active_cats and cls_id < len(active_cats))
                              else str(cls_id))
                label_text = f"{cls_name} {scores[i]:.2f}"
                valid_dets.append((x1, y1, x2, y2, cls_id, color, label_text))
                all_bboxes.append((x1, y1, x2, y2))

            # 第二遍：為每個偵測智慧放置標籤（避開所有其他框與已放標籤）
            placed_labels = []

            for x1, y1, x2, y2, cls_id, color, label_text in valid_dets:
                (tw, th), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                pad = 4
                lw  = tw + pad

                # 候選位置 (lx, text_y)：text_y 為 putText baseline Y
                # 優先：框上方 → 框下方 → 框內上方 → 框內下方；各嘗試靠左/靠右
                candidates = []
                for lx_base in [x1, max(0, x2 - lw)]:
                    lx = max(0, min(lx_base, orig_w - lw))
                    candidates.append((lx, y1 - pad))
                    candidates.append((lx, y2 + th + pad))
                    candidates.append((lx, y1 + th + pad))
                    candidates.append((lx, max(th + pad, y2 - baseline - pad)))

                # 禁止區 = 其他所有框 + 已放標籤（排除自身框）
                others = [b for b in all_bboxes
                          if not (b[0] == x1 and b[1] == y1
                                  and b[2] == x2 and b[3] == y2)
                         ] + placed_labels

                chosen_lx, chosen_ty, chosen_rect = None, None, None
                for lx, text_y in candidates:
                    text_y = max(th + pad, min(text_y, orig_h - baseline - pad))
                    lr = (lx,
                          text_y - th - pad // 2,
                          lx + lw,
                          text_y + baseline + pad // 2)
                    if not any(_overlaps(lr, o) for o in others):
                        chosen_lx, chosen_ty, chosen_rect = lx, text_y, lr
                        break

                # fallback：框正上方，不管碰撞
                if chosen_lx is None:
                    chosen_lx = max(0, min(x1, orig_w - lw))
                    chosen_ty = max(th + pad, y1 - pad)
                    chosen_rect = (chosen_lx,
                                   chosen_ty - th - pad // 2,
                                   chosen_lx + lw,
                                   chosen_ty + baseline + pad // 2)

                cv2.rectangle(vis_img,
                              (chosen_lx, chosen_rect[1]),
                              (chosen_lx + lw, chosen_rect[3]),
                              color, -1)
                cv2.putText(vis_img, label_text,
                            (chosen_lx + pad // 2, chosen_ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                placed_labels.append(chosen_rect)

            cv2.imwrite(os.path.join(save_dir, f"{img_stem}.png"), vis_img)

        # ────────────────────────────────────────────────────────────
        # 任務 B：RM（語意分割）
        # ────────────────────────────────────────────────────────────
        elif args.task == 'rm':
            views = rm_views if rm_views else [input_tensor]
            all_logits = [
                slide_inference_rm(
                    model, t,
                    num_classes=segconfig_rm.num_labels,
                    crop_size=tuple(cfg.crop_size),
                    stride=tuple(cfg.stride),
                    device=device,
                )
                for t in views
            ]
            logits = torch.stack(all_logits).mean(0)
            logits = TF.resize(logits, (orig_h, orig_w),
                               interpolation=InterpolationMode.BILINEAR)
            pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            pred_mask = rm_post_process(pred_mask, min_area=int(orig_h * orig_w * 0.0003))

            if palette is not None:
                color_mask     = palette[pred_mask]
                color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
                original_img   = cv2.imread(img_path)
                if original_img.shape[:2] != color_mask_bgr.shape[:2]:
                    color_mask_bgr = cv2.resize(
                        color_mask_bgr, (orig_w, orig_h))
                vis_result = cv2.addWeighted(
                    original_img, 1 - args.opacity,
                    color_mask_bgr, args.opacity, 0)
            else:
                vis_result = pred_mask * 50

            cv2.imwrite(os.path.join(save_dir, f"{img_stem}.png"), vis_result)

        # ────────────────────────────────────────────────────────────
        # 任務 C：LL（車道線偵測 - 純語意分割版）
        # ────────────────────────────────────────────────────────────
        elif args.task == 'll':
            mask_logits  = outputs.get('mask_logits', outputs.get('logits'))
            original_img = cv2.imread(img_path)
            vis_result   = original_img.copy()
 
            if mask_logits is None:
                cv2.imwrite(os.path.join(save_dir, f"{img_stem}.png"), vis_result)
                continue
 
            # ── 1. Resize logits → original size ────────────────────
            mask_logits = TF.resize(
                mask_logits, (orig_h, orig_w),
                interpolation=InterpolationMode.BILINEAR
            )
 
            # ── 2. Sigmoid + threshold ───────────────────────────────
            prob      = torch.sigmoid(mask_logits.squeeze()).cpu().numpy()
            bin_mask  = (prob > args.ll_thresh).astype(np.uint8)  # {0,1}
 
            if not bin_mask.any():
                cv2.imwrite(os.path.join(save_dir, f"{img_stem}.png"), vis_result)
                continue
 
            # ── 3. Morphological clean-up ────────────────────────────
            # 儲存膨脹前的 mask，後續用來量原始 Y 跨度（不受膨脹膨脹影響）
            bin_mask_orig = bin_mask.copy()
            # 垂直膨脹橋接虛線段間空隙
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            bin_mask = cv2.dilate(bin_mask, kernel_v, iterations=1)
            # Close 填補微小斷裂
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            bin_mask = cv2.morphologyEx(
                bin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

            # ── 4. Connected components ───────────────────────────────
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                bin_mask, connectivity=8)
            # 依解析度縮放：720p→720px，避免將小雜訊 component 當成車道
            min_pixels = max(80, orig_h)
            # 原始 mask（膨脹前）Y 跨度門檻：4% 圖高（720p ≈ 29px）
            min_orig_height = max(20, int(orig_h * 0.04))

            # ── 5a. 收集通過過濾的 component ────────────────────────
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

                # 若 X 跨度過大先拆分（兩條車道在 mask 上相連）
                for sub_xs, sub_ys in _maybe_split_component(xs, ys, orig_w):
                    if len(sub_ys) >= 20:
                        raw_components.append((sub_xs, sub_ys))

            # ── 5b. 合併同一虛線車道的相鄰段 ────────────────────────
            grouped = _group_nearby_components(raw_components, orig_h, orig_w)

            # ── 5c. PCA 擬合每個群組 ──────────────────────────────
            lane_curves = []
            min_curve_len = max(50, orig_h * 0.08)   # 5% 圖高，720p ≈ 36px
            for g_xs, g_ys in grouped:
                pts = _lane_fit_pca(g_xs, g_ys, orig_h, orig_w)
                if pts is None:
                    continue
                # 弧長過短 → 視為雜訊，跳過
                diffs = np.diff(pts.astype(float), axis=0)
                arc_len = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
                if arc_len < min_curve_len:
                    continue
                lane_curves.append((float(np.median(g_xs)), pts))

            lane_curves.sort(key=lambda t: t[0])
            lane_curves = [pts for _, pts in lane_curves]
 
            # ── 6. 畫車道線 ──────────────────────────────────────────
            # 使用不同顏色區分不同車道（最多 6 條）
            COLORS = [
                (  0, 255,   0),   # 綠
                (  0, 200, 255),   # 青
                (255, 100,   0),   # 橙藍
                (255,   0, 255),   # 洋紅
                (  0, 100, 255),   # 藍橙
                (100, 255, 100),   # 淡綠
            ]
            LINE_WIDTH = max(3, orig_h // 170)  # 依解析度自動調整線寬
 
            overlay = vis_result.copy()
            for i, pts in enumerate(lane_curves):
                color = COLORS[i % len(COLORS)]
                cv2.polylines(
                    overlay,
                    [pts.reshape(-1, 1, 2)],
                    isClosed=False,
                    color=color,
                    thickness=LINE_WIDTH,
                    lineType=cv2.LINE_AA   # anti-aliased → 平滑無鋸齒
                )
 
            # 半透明疊加
            vis_result = cv2.addWeighted(vis_result, 0.4, overlay, 0.6, 0)
            cv2.imwrite(os.path.join(save_dir, f"{img_stem}.png"), vis_result)
        # ────────────────────────────────────────────────────────────
        # 任務 C-2：LLSEG（車道線純語意分割遮罩輸出）
        # ────────────────────────────────────────────────────────────
        elif args.task == 'llseg':
            mask_logits  = outputs.get('mask_logits', outputs.get('logits'))
            original_img = cv2.imread(img_path)
            
            if mask_logits is None:
                print(f"[Warning] No mask_logits found for {img_stem}")
                cv2.imwrite(os.path.join(save_dir, f"{img_stem}.png"), original_img)
                continue
 
            # ── 1. Resize logits 放大回原圖尺寸 ───────────────────────
            mask_logits = TF.resize(
                mask_logits, (orig_h, orig_w),
                interpolation=InterpolationMode.BILINEAR
            )
 
            # ── 2. 經過 Sigmoid 轉成機率值 [0, 1] ───────────────────
            prob = torch.sigmoid(mask_logits.squeeze()).cpu().numpy()
 
            # ── 3. 二值化 Mask (大於閾值的視為車道線前景) ─────────────
            bin_mask = (prob > args.ll_thresh).astype(np.uint8)
 
            # ── 4. 繪製純色遮罩 ─────────────────────────────────────
            # 建立一個與原圖大小相同的綠色畫布 (BGR)
            color_mask = np.zeros_like(original_img)
            color_mask[bin_mask == 1] = [0, 255, 0]  # 螢光綠
            
            vis_result = original_img.copy()
            
            # ── 5. 將 Mask 半透明疊加回原圖 ─────────────────────────
            # 為了畫面乾淨，我們只在「模型預測為車道線」的區域進行疊加
            lane_region = bin_mask == 1
            if lane_region.any():
                alpha = 0.5  # 遮罩透明度，可自行調整
                vis_result[lane_region] = cv2.addWeighted(
                    original_img[lane_region], 1 - alpha,
                    color_mask[lane_region], alpha, 0
                )
 
            # ── 6. (選用) 在邊緣描一層白邊讓視覺更銳利 ───────────────
            contours, _ = cv2.findContours(
                bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_result, contours, -1,
                             color=(255, 255, 255), thickness=1,
                             lineType=cv2.LINE_AA)
 
            cv2.imwrite(os.path.join(save_dir, f"{img_stem}.png"), vis_result)
            
    print("\nDone!")


if __name__ == "__main__":
    main()