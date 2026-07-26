"""
tools/make_demo_video.py
────────────────────────────────────────────────────────────────────────
把一個資料夾內的行車影片幀（已裁切好的圖片），逐幀同時餵給
MultiTaskSegFormer 的四個 head（RM / LL / TS / TL），將四個結果排成
2×2 田字格，合成一段展示影片 (mp4)。

視覺化邏輯完全沿用 tools/inference_multitask.py 的 helper，確保與
單任務 inference 一致，不會分岔。

排版：
    ┌──────────────┬──────────────┐
    │ 道路標線 RM   │ 車道線 LL     │   ← 語意分割 / 曲線疊圖
    ├──────────────┼──────────────┤
    │ 交通標誌 TS   │ 紅綠燈 TL     │   ← YOLOX 偵測框
    └──────────────┴──────────────┘

用法：
    python -m tools.make_demo_video \
        --config configs/train_uda_multi_tasks.json \
        --checkpoint path/to/ckpt.pth \
        --input  path/to/frames_dir \
        --output demo.mp4 \
        --fps 10

    # 只想先看逐幀 PNG（不寫影片）：加 --save_frames frames_out/
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import SegformerConfig

from configs.multitask_config import MultiTaskTrainingConfig
from engine.category import Category
from engine.decode_utils import decode_yolox_outputs
from engine.multi_task_segformer import get_model as get_mt_model

# 直接重用 inference 的 helper（純函式，import 不會觸發 argparse/main）
from tools.inference_multitask import (
    slide_inference_rm,
    rm_post_process,
    _lane_fit_pca,
    _group_nearby_components,
    _maybe_split_component,
    preprocess_image,
    get_palette,
)

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# =================================================================================
# 四個任務的單幀渲染：各自回傳「原圖解析度」的 BGR ndarray
# =================================================================================
def render_rm(model, img_path, cfg, palette, num_labels, device,
              opacity=0.6, conf_thresh=0.45):
    """道路標線語意分割：彩色 mask 半透明疊圖。回傳 BGR 原圖尺寸。"""
    target_h, target_w = cfg.image_scale
    tensor, (orig_h, orig_w), _ = preprocess_image(img_path, target_h, target_w)
    original_img = cv2.imread(img_path)
    if tensor is None:
        return original_img
    tensor = tensor.to(device)

    logits = slide_inference_rm(
        model, tensor, num_classes=num_labels,
        crop_size=tuple(cfg.crop_size), stride=tuple(cfg.stride), device=device,
    )
    logits = TF.resize(logits, (orig_h, orig_w),
                       interpolation=InterpolationMode.BILINEAR)
    probs = torch.softmax(logits, dim=1)
    max_prob, pred_idx = probs.max(dim=1)
    pred_idx[(pred_idx > 0) & (max_prob < conf_thresh)] = 0
    pred_mask = pred_idx.squeeze(0).cpu().numpy().astype(np.uint8)
    pred_mask = rm_post_process(pred_mask, min_area=int(orig_h * orig_w * 0.0006))

    if palette is None:
        return original_img
    color_mask = cv2.cvtColor(palette[pred_mask], cv2.COLOR_RGB2BGR)
    if color_mask.shape[:2] != (orig_h, orig_w):
        color_mask = cv2.resize(color_mask, (orig_w, orig_h))
    return cv2.addWeighted(original_img, 1 - opacity, color_mask, opacity, 0)


def render_ll(model, img_path, cfg, device, ll_thresh=0.4):
    """車道線：分割 → 連通分量 → PCA 擬合曲線疊圖。回傳 BGR 原圖尺寸。"""
    target_h, target_w = cfg.crop_size
    tensor, (orig_h, orig_w), _ = preprocess_image(img_path, target_h, target_w)
    original_img = cv2.imread(img_path)
    if tensor is None:
        return original_img
    tensor = tensor.to(device)

    autocast_enabled = (device.type == "cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=autocast_enabled):
        out = model(pixel_values=tensor, task="ll")
    mask_logits = out.get("mask_logits", out.get("logits"))
    if mask_logits is None:
        return original_img

    mask_logits = TF.resize(mask_logits, (orig_h, orig_w),
                            interpolation=InterpolationMode.BILINEAR)
    prob = torch.sigmoid(mask_logits.squeeze()).cpu().numpy()
    bin_mask = (prob > ll_thresh).astype(np.uint8)
    if not bin_mask.any():
        return original_img

    bin_mask_orig = bin_mask.copy()
    bin_mask = cv2.dilate(
        bin_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)), iterations=1)
    bin_mask = cv2.morphologyEx(
        bin_mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    min_pixels = max(80, orig_h)
    min_orig_height = max(20, int(orig_h * 0.04))

    raw_components = []
    for lab in range(1, n_labels):
        if stats[lab, cv2.CC_STAT_AREA] < min_pixels:
            continue
        ys, xs = np.where(labels == lab)
        if len(ys) < 20:
            continue
        orig_ys = np.where((labels == lab) & (bin_mask_orig == 1))[0]
        if len(orig_ys) == 0 or orig_ys.max() - orig_ys.min() < min_orig_height:
            continue
        for sub_xs, sub_ys in _maybe_split_component(xs, ys, orig_w):
            if len(sub_ys) >= 20:
                raw_components.append((sub_xs, sub_ys))

    grouped = _group_nearby_components(raw_components, orig_h, orig_w)

    lane_curves = []
    min_curve_len = max(50, orig_h * 0.08)
    for g_xs, g_ys in grouped:
        pts = _lane_fit_pca(g_xs, g_ys, orig_h, orig_w)
        if pts is None:
            continue
        diffs = np.diff(pts.astype(float), axis=0)
        if float(np.sum(np.sqrt((diffs ** 2).sum(axis=1)))) < min_curve_len:
            continue
        lane_curves.append((float(np.median(g_xs)), pts))
    lane_curves.sort(key=lambda t: t[0])
    lane_curves = [pts for _, pts in lane_curves]

    COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0),
              (255, 0, 255), (0, 100, 255), (100, 255, 100)]
    line_width = max(3, orig_h // 170)
    overlay = original_img.copy()
    for i, pts in enumerate(lane_curves):
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], isClosed=False,
                      color=COLORS[i % len(COLORS)], thickness=line_width,
                      lineType=cv2.LINE_AA)
    return cv2.addWeighted(original_img, 0.4, overlay, 0.6, 0)


def render_det(model, img_path, cfg, task, cats, palette, device, conf_thresh):
    """交通標誌 / 紅綠燈偵測：畫 bbox + 標籤。回傳 BGR 原圖尺寸。

    線寬與字體依原圖寬度縮放，確保縮到田字格小格後仍看得清楚。
    """
    tensor, (orig_h, orig_w), (res_h, res_w) = preprocess_image(img_path, 960, 960)
    vis_img = cv2.imread(img_path)
    if tensor is None:
        return vis_img
    tensor = tensor.to(device)
    scale_x, scale_y = orig_w / res_w, orig_h / res_h

    autocast_enabled = (device.type == "cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=autocast_enabled):
        outputs = model(pixel_values=tensor, task=task)
    final_bboxes, final_scores, final_classes = decode_yolox_outputs(
        outputs["logits"], conf_thresh=conf_thresh, nms_thresh=0.45)

    bboxes = final_bboxes[0].cpu().numpy() if (final_bboxes and final_bboxes[0].numel() > 0) else []
    scores = final_scores[0].cpu().numpy() if (final_scores and final_scores[0].numel() > 0) else []
    clses = final_classes[0].cpu().numpy() if (final_classes and final_classes[0].numel() > 0) else []

    # 依原圖寬度縮放繪圖參數（田字格內格寬約 640 → s≈orig_w/640）
    s = max(1.0, orig_w / 640.0)
    box_lw = max(2, int(round(2 * s)))
    font_scale = 0.45 * s
    font_th = max(1, int(round(s)))

    for i in range(len(scores)):
        x1 = max(0, int(bboxes[i][0] * scale_x)); y1 = max(0, int(bboxes[i][1] * scale_y))
        x2 = min(orig_w, int(bboxes[i][2] * scale_x)); y2 = min(orig_h, int(bboxes[i][3] * scale_y))
        cls_id = int(clses[i])

        # TL 車燈誤判過濾（沿用 inference 邏輯）
        if task == "tl":
            box_w = x2 - x1
            box_h = max(y2 - y1, 1)
            box_cy = (y1 + y2) / 2
            if box_cy > orig_h * 0.65 and box_w > orig_w * 0.10:
                continue
            if box_w / box_h > 2.5:
                continue
            if cls_id == 2 and scores[i] < max(conf_thresh, 0.55):
                continue

        if palette is not None and cls_id < len(palette):
            c = palette[cls_id]
            color = (int(c[2]), int(c[1]), int(c[0]))
        else:
            color = (0, 255, 0)

        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, box_lw)
        cls_name = cats[cls_id].name if (cats and cls_id < len(cats)) else str(cls_id)
        label = f"{cls_name} {scores[i]:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_th)
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(vis_img, (x1, ty - th - 4), (x1 + tw + 4, ty + base), color, -1)
        cv2.putText(vis_img, label, (x1 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_th)
    return vis_img


# =================================================================================
# 田字格合成
# =================================================================================
def fit_into(img, w, h):
    """等比縮放 + 置中黑邊（letterbox），不變形。"""
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def make_cell(img, title, cell_w, inner_h, bar_h, bar_color):
    """單格 = 標題列 + 內容圖。"""
    cell = np.zeros((bar_h + inner_h, cell_w, 3), np.uint8)
    # 標題列
    cell[:bar_h] = bar_color
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(cell, title, (10, (bar_h + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    # 內容
    cell[bar_h:] = fit_into(img, cell_w, inner_h)
    return cell


def compose_grid(panels, titles, colors, cell_w, inner_h, bar_h):
    """panels/titles/colors 依序為 [RM, LL, TS, TL]，排成 2×2。"""
    cells = [make_cell(p, t, cell_w, inner_h, bar_h, c)
             for p, t, c in zip(panels, titles, colors)]
    top = np.hstack([cells[0], cells[1]])
    bot = np.hstack([cells[2], cells[3]])
    # 中間分隔線
    frame = np.vstack([top, bot])
    return frame


# =================================================================================
# 主程式
# =================================================================================
def build_model(cfg, device):
    seg_rm = SegformerConfig.from_pretrained("nvidia/mit-b1")
    seg_rm.num_labels = len(Category.load(cfg.category_csv_rlmd))
    seg_ll = SegformerConfig.from_pretrained("nvidia/mit-b1")
    seg_ll.num_labels = len(Category.load(cfg.category_csv_ll))

    cats_ts = Category.load(cfg.category_csv_ts) if (
        getattr(cfg, "category_csv_ts", None) and os.path.exists(cfg.category_csv_ts)) else []
    seg_ts = SegformerConfig.from_pretrained("nvidia/mit-b1")
    seg_ts.num_labels = len(cats_ts) if cats_ts else getattr(cfg, "ts_num_classes", 13)

    cats_tl = Category.load(cfg.category_csv_tl) if (
        getattr(cfg, "category_csv_tl", None) and os.path.exists(cfg.category_csv_tl)) else []
    seg_tl = SegformerConfig.from_pretrained("nvidia/mit-b1")
    seg_tl.num_labels = len(cats_tl) if cats_tl else getattr(cfg, "tl_num_classes", 4)

    model = get_mt_model(seg_rm, seg_ll, seg_ts, seg_tl)
    return model, seg_rm.num_labels, cats_ts, cats_tl


def parse_args():
    p = argparse.ArgumentParser(description="Make 2x2 multi-task demo video")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="資料夾：內含依序命名的影片幀")
    p.add_argument("--output", default="demo.mp4", help="輸出 mp4 路徑")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cell_w", type=int, default=640, help="每格寬")
    p.add_argument("--inner_h", type=int, default=340, help="每格內容高（不含標題列）")
    p.add_argument("--bar_h", type=int, default=36, help="標題列高")
    p.add_argument("--save_frames", default=None, help="另存逐幀合成 PNG 的資料夾")
    p.add_argument("--rm_opacity", type=float, default=0.6)
    p.add_argument("--rm_conf_thresh", type=float, default=0.45)
    p.add_argument("--ll_thresh", type=float, default=0.4)
    p.add_argument("--ts_conf_thresh", type=float, default=0.5)
    p.add_argument("--tl_conf_thresh", type=float, default=0.3)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = MultiTaskTrainingConfig.load(args.config)
    print(f"Loaded config: {args.config}")

    # 色盤
    palette_rm = get_palette(cfg.category_csv_rlmd)
    palette_ts = get_palette(getattr(cfg, "category_csv_ts", None))
    palette_tl = get_palette(getattr(cfg, "category_csv_tl", None))

    # 模型
    print("Building model...")
    model, rm_labels, cats_ts, cats_tl = build_model(cfg, device)
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()
                if "discriminator" not in k}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        print(f"[Checkpoint] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[Checkpoint] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    model.to(device).eval()

    # 影格清單
    if not os.path.isdir(args.input):
        raise SystemExit(f"--input 必須是資料夾: {args.input}")
    frames = sorted(os.path.join(args.input, f) for f in os.listdir(args.input)
                    if os.path.splitext(f)[1].lower() in VALID_EXT)
    if not frames:
        raise SystemExit(f"資料夾內沒有圖片: {args.input}")
    print(f"Found {len(frames)} frames.")

    # 田字格：標題與配色（BGR）— 分割類暖色、偵測類冷色
    titles = ["Road Marking (RM)", "Lane Line (LL)",
              "Traffic Sign (TS)", "Traffic Light (TL)"]
    colors = [(40, 90, 200), (40, 150, 90), (150, 90, 40), (140, 60, 120)]

    cell_w, inner_h, bar_h = args.cell_w, args.inner_h, args.bar_h
    frame_w, frame_h = cell_w * 2, (bar_h + inner_h) * 2

    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, args.fps, (frame_w, frame_h))
        if not writer.isOpened():
            raise SystemExit("無法開啟 VideoWriter，請確認 ffmpeg/opencv 編碼器可用。")
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    print(f"Composing {frame_w}x{frame_h} @ {args.fps}fps -> {args.output}")
    for img_path in tqdm(frames):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        rm_p = render_rm(model, img_path, cfg, palette_rm, rm_labels, device,
                         opacity=args.rm_opacity, conf_thresh=args.rm_conf_thresh)
        ll_p = render_ll(model, img_path, cfg, device, ll_thresh=args.ll_thresh)
        ts_p = render_det(model, img_path, cfg, "ts", cats_ts, palette_ts, device,
                          conf_thresh=args.ts_conf_thresh)
        tl_p = render_det(model, img_path, cfg, "tl", cats_tl, palette_tl, device,
                          conf_thresh=args.tl_conf_thresh)

        frame = compose_grid([rm_p, ll_p, ts_p, tl_p], titles, colors,
                             cell_w, inner_h, bar_h)
        if writer is not None:
            writer.write(frame)
        if args.save_frames:
            cv2.imwrite(os.path.join(args.save_frames, f"{stem}.png"), frame)

    if writer is not None:
        writer.release()
    print(f"\nDone! -> {args.output}")


if __name__ == "__main__":
    main()
