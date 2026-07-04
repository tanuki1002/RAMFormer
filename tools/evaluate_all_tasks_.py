# evaluate_all_tasks.py
# 可以使用 --task 參數來指定要評估的任務（預設為 all）
# 例如： python evaluate_all_tasks.py --config configs/train_uda_multi_tasks.json --checkpoint logs/你的實驗檔/best_model.pth --task ll
# 評估所有任務 python evaluate_all_tasks.py --config configs/train_uda_multi_tasks.json --checkpoint logs/你的實驗檔/best_model.pth --task all
# 或是直接省略 --task，預設就是 all

import argparse
import os
import torch
import math
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from transformers import SegformerConfig
from tqdm import tqdm
import random

# Engine modules
from engine.multi_task_segformer import get_model as get_mt_model
from engine.dataloader import get_dataset
from engine.category import Category
from engine import transform
from engine.metric import Metrics
from engine.validator import Validator
from engine.decode_utils import decode_yolox_outputs
from configs.multitask_config import MultiTaskTrainingConfig

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

# =================================================================================
# 1. 輔助與轉換類別
# =================================================================================
class TqdmDataLoader:
    def __init__(self, dataloader, desc=""):
        self.dataloader = dataloader
        self.desc = desc
        
    def __iter__(self):
        return iter(tqdm(self.dataloader, desc=self.desc, dynamic_ncols=True))
        
    def __len__(self):
        return len(self.dataloader)

class LoadBDDColorLabelToID:
    def transform(self, data):
        if data.get("ann_path") is None:
            if "img" in data:
                h, w = data["img"].shape[-2:] if isinstance(data["img"], torch.Tensor) else data["img"].shape[:2]
            else:
                h, w = 512, 512
            data["ann"] = torch.full((1, h, w), 255, dtype=torch.long)
            return data
        
        ann = Image.open(data["ann_path"])
        ann_np = np.asarray(ann).copy()
        id_map = np.zeros(ann_np.shape[:2], dtype=np.int64)

        if len(ann_np.shape) == 3:
            mask_foreground = (ann_np[:, :, 0] > 100) | (ann_np[:, :, 1] > 100) | (ann_np[:, :, 2] > 100)
            id_map[mask_foreground] = 1
        else:
            mask_255 = (ann_np == 255)
            id_map[mask_255] = 1

        data["ann"] = torch.from_numpy(id_map)[None, :].long()
        return data

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
            
            # [CRITICAL FIX] Segformer 輸出為 1/4 解析度。
            # 必須在這裡將 Logits 放大回輸入圖片的尺寸，
            # 否則 Validator 在進行 Sliding Window 拼接時會導致畫面嚴重碎裂與錯位！
            if logits is not None and logits.shape[-2:] != img_input.shape[-2:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=img_input.shape[-2:], mode='bilinear', align_corners=False
                )
                
            return logits, outputs.get('loss', None)
        return outputs

# =================================================================================
# 2. 車道線 (Lane Line) 30px F1-Score 評估邏輯 (推論預測對齊版)
# =================================================================================

# ── 以下三個 helper 完全對齊 inference_multitask.py ─────────────────────────────

def _ll_lane_fit_pca(xs, ys, orig_h, orig_w):
    """PCA 主軸方向擬合，與 inference_multitask._lane_fit_pca 完全一致。"""
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


def _ll_maybe_split_component(xs, ys, orig_w):
    """X 直方圖谷點拆分寬大 blob，與 inference_multitask._maybe_split_component 一致。"""
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


def _ll_group_nearby_components(components, orig_h, orig_w):
    """方向延伸合併虛線段，與 inference_multitask._group_nearby_components 一致。"""
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


def get_pred_30px_regions(bin_mask, lane_width=30, min_pixels=None):
    """
    [預測端專用] 完全對齊 inference_multitask.py 的後處理邏輯：
    垂直膨脹 + 閉運算 → CC 過濾 → 拆分寬大 blob → 合併虛線段（方向延伸）
    → PCA 擬合 → 弧長過濾 → 畫成指定寬度的平滑實線
    """
    H, W = bin_mask.shape

    if min_pixels is None:
        min_pixels = max(80, H)

    bin_mask_orig = bin_mask.copy()

    kernel_v     = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    bin_mask     = cv2.dilate(bin_mask, kernel_v, iterations=1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_mask  = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

    min_orig_height = max(20, int(H * 0.04))   # 對齊 inference_multitask.py（4%）

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

        for sub_xs, sub_ys in _ll_maybe_split_component(xs, ys, W):
            if len(sub_ys) >= 20:
                raw_components.append((sub_xs, sub_ys))

    grouped = _ll_group_nearby_components(raw_components, H, W)

    min_curve_len = max(50, H * 0.05)
    regions = []
    for g_xs, g_ys in grouped:
        pts = _ll_lane_fit_pca(g_xs, g_ys, H, W)
        if pts is None:
            continue
        diffs   = np.diff(pts.astype(float), axis=0)
        arc_len = float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
        if arc_len < min_curve_len:
            continue
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)],
                      isClosed=False, color=1, thickness=lane_width)
        regions.append(mask)

    return regions

def get_gt_30px_regions(bin_mask, lane_width=30, min_pixels=15):
    """
    [Ground Truth 專用 - 距離轉換自適應膨脹版]
    完全不使用 polyfit，也絕對不使用座標平均！
    利用 Distance Transform 自動計算原始 GT 的粗度，只補足缺少的寬度，
    完美保持 Ground Truth 最原始的彎曲與拓撲形狀。
    """
    # 1. 垂直膨脹連接虛線 (僅用於連通集分群，不影響最終畫線的形狀)
    kernel_v = np.ones((40, 3), np.uint8)
    closed = cv2.dilate(bin_mask, kernel_v, iterations=1)
    
    # 2. 區分不同的車道線實例
    n_labels, labels = cv2.connectedComponents(closed, connectivity=8)
    
    regions = []
    for lbl in range(1, n_labels):
        # 3. 提取原始最乾淨的 GT 像素，作為長胖的基底
        comp = ((labels == lbl) & (bin_mask > 0)).astype(np.uint8)
        
        if comp.sum() < min_pixels:
            continue
            
        # 4. 神級技巧：使用距離轉換計算該條線目前的「最大半徑」
        # cv2.DIST_L2 代表歐氏距離，可以精準算出中心點到邊緣的像素距離
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        max_radius = np.max(dist)
        current_width = max_radius * 2  # 算出這條 GT 線當前的真實寬度
        
        # 5. 計算需要補足的膨脹大小 (目標寬度 - 當前寬度)
        # 例如：目標 30px，當前 GT 是 12px，那我們只需要膨脹 18px
        dilate_size = int(max(1, lane_width - current_width))
        
        if dilate_size > 1:
            # 使用橢圓(圓形) Kernel，確保向外擴張的邊緣是平滑的
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            thick_mask = cv2.dilate(comp, kernel, iterations=1)
        else:
            thick_mask = comp
            
        regions.append(thick_mask)
        
    return regions

class LaneLineEvaluator:
    def __init__(self, thresholds=[0.3, 0.5], thickness=30, prob_thresh=0.5):
        self.thresholds = thresholds
        self.thickness = thickness
        self.prob_thresh = prob_thresh
        self.results = {t: {'TP': 0, 'FP': 0, 'FN': 0} for t in thresholds}
        self.total_inter   = 0
        self.total_union   = 0
        self.total_correct = 0   # TP + TN（正確分類的像素數）
        self.total_pixels  = 0   # 總像素數

    def update(self, pred_logits, gt_mask):
        B, _, H, W = pred_logits.shape
        pred_probs = torch.sigmoid(pred_logits).squeeze(1).cpu().numpy()

        if gt_mask.dim() == 4:
            gt_masks_np = gt_mask[:, 0, :, :].cpu().numpy()
        else:
            gt_masks_np = gt_mask.cpu().numpy()

        for b in range(B):
            p_bin = (pred_probs[b] > self.prob_thresh).astype(np.uint8)

            # lane_mask 值為 0.0 / 1.0 (float32)
            g_bin = (gt_masks_np[b] > 0.5).astype(np.uint8)

            self.total_inter   += np.logical_and(p_bin, g_bin).sum()
            self.total_union   += np.logical_or(p_bin, g_bin).sum()
            # Accuracy：預測與 GT 完全相同的像素（含背景 TN）
            self.total_correct += np.sum(p_bin == g_bin)
            self.total_pixels  += p_bin.size
            
            # [雙軌制擷取] 預測端用擬合畫線，GT端用形態學擴張
            p_regions = get_pred_30px_regions(p_bin, lane_width=self.thickness)
            g_regions = get_gt_30px_regions(g_bin, lane_width=self.thickness, min_pixels=15)
            
            iou_mat = np.zeros((len(p_regions), len(g_regions)))
            for i, pm in enumerate(p_regions):
                for j, gm in enumerate(g_regions):
                    inter = np.logical_and(pm, gm).sum()
                    union = np.logical_or(pm, gm).sum()
                    iou_mat[i, j] = inter / (union + 1e-8)

            # 依各預測區域的平均信心分數由高到低排列，確保貪婪匹配優先分配高信心預測
            if len(p_regions) > 1:
                p_conf = np.array([
                    float(pred_probs[b][pm == 1].mean()) if pm.sum() > 0 else 0.0
                    for pm in p_regions
                ])
                iou_mat = iou_mat[np.argsort(p_conf)[::-1]]

            for thresh in self.thresholds:
                tp = 0
                matched_gt = set()
                for i in range(len(p_regions)):
                    if len(g_regions) == 0: break
                    best_gt, best_iou = -1, -1
                    for j in range(len(g_regions)):
                        if j in matched_gt: continue
                        if iou_mat[i, j] > best_iou:
                            best_iou = iou_mat[i, j]
                            best_gt = j
                    if best_iou >= thresh:
                        tp += 1
                        matched_gt.add(best_gt)
                        
                fp = len(p_regions) - tp
                fn = len(g_regions) - len(matched_gt)
                
                self.results[thresh]['TP'] += tp
                self.results[thresh]['FP'] += fp
                self.results[thresh]['FN'] += fn

    def compute(self):
        metrics = {}
        for t in self.thresholds:
            tp = self.results[t]['TP']
            fp = self.results[t]['FP']
            fn = self.results[t]['FN']
            eps = 1e-8
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            
            metrics[f'F1@{t}'] = f1
            metrics[f'P@{t}'] = precision
            metrics[f'R@{t}'] = recall

        metrics['Pixel_IoU'] = self.total_inter / (self.total_union + 1e-8)
        metrics['Accuracy']  = self.total_correct / (self.total_pixels + 1e-8)
        return metrics

def evaluate_ll_task(name, model, dataloader, device, mode_name="Source"):
    if dataloader is None: return None
    print(f"\n[{name.upper()}] Evaluating {mode_name} Validation Set (30px Instance F1 & Pixel IoU)...")
    evaluator = LaneLineEvaluator(thresholds=[0.3, 0.5], thickness=30, prob_thresh=0.3)
    model.eval()
    autocast_enabled = (device.type == 'cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=autocast_enabled):
        for data in tqdm(dataloader, desc=f"  {name.upper()} {mode_name}", dynamic_ncols=True):
            imgs = data['img'].to(device)
            
            if 'lane_mask' in data:
                gt_mask = data['lane_mask'].to(device)
            elif 'ann' in data:
                gt_mask = data['ann'].to(device)
            else:
                continue
                
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.unsqueeze(1)
            
            outputs = model(pixel_values=imgs, task='ll')
            mask_logits = outputs.get('mask_logits', outputs.get('logits'))

            if mask_logits.shape[-2:] != gt_mask.shape[-2:]:
                gt_mask = torch.nn.functional.interpolate(
                    gt_mask.float(), size=mask_logits.shape[-2:], mode='nearest'
                )

            evaluator.update(mask_logits, gt_mask)
            
    res = evaluator.compute()
    print(f"  -> {mode_name} Pixel IoU: {res['Pixel_IoU']:.4f} | Accuracy: {res['Accuracy']:.4f} | F1@0.5: {res['F1@0.5']:.4f} | F1@0.3: {res['F1@0.3']:.4f}")
    return res

def test_lane(model, dataloader, device, num_samples=5, save_dir="debug_lanes_30px"):
    if dataloader is None: return
    print(f"\n[DEBUG] 正在隨機抽取 {num_samples} 張車道線進行 30px 視覺化驗證...")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 隨機挑 batch index，迭代時跳過不需要的 batch，避免整個驗證集 OOM
    total_batches = len(dataloader)
    selected = set(random.sample(range(total_batches), min(num_samples, total_batches)))

    count = 0
    autocast_enabled = (device.type == 'cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=autocast_enabled):
        for batch_idx, data in enumerate(dataloader):
            if batch_idx not in selected:
                continue
            imgs = data['img'].to(device)

            if 'lane_mask' in data:
                gt_masks = data['lane_mask'].to(device)
            elif 'ann' in data:
                gt_masks = data['ann'].to(device)
            else:
                continue

            if gt_masks.dim() == 3:
                gt_masks = gt_masks.unsqueeze(1)

            outputs = model(pixel_values=imgs, task='ll')
            mask_logits = outputs.get('mask_logits', outputs.get('logits'))
            
            if mask_logits.shape[-2:] != gt_masks.shape[-2:]:
                gt_masks = torch.nn.functional.interpolate(
                    gt_masks.float(), size=mask_logits.shape[-2:], mode='nearest'
                )
            
            pred_probs = torch.sigmoid(mask_logits).squeeze(1).cpu().numpy()
            gt_masks_np = gt_masks.squeeze(1).cpu().numpy()
            imgs_np = imgs.cpu().numpy()
            
            B = imgs.shape[0]
            for b in range(B):
                if count >= num_samples:
                    print(f" 成功儲存 {num_samples} 張測試對照圖至 `{save_dir}/` 目錄下！")
                    return
                
                p_bin = (pred_probs[b] > 0.3).astype(np.uint8)
                g_bin = (gt_masks_np[b] > 0.5).astype(np.uint8)
                
                # 視覺化也切換為雙軌制
                p_regions = get_pred_30px_regions(p_bin, lane_width=30)
                g_regions = get_gt_30px_regions(g_bin, lane_width=30, min_pixels=15)
                
                img = imgs_np[b].transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                gt_color = np.zeros_like(img_bgr)
                pred_color = np.zeros_like(img_bgr)
                
                for gm in g_regions:
                    gt_color[gm == 1] = [0, 0, 255] 
                for pm in p_regions:
                    pred_color[pm == 1] = [0, 255, 0]
                    
                vis_gt = cv2.addWeighted(img_bgr, 0.7, gt_color, 0.6, 0)
                vis_pred = cv2.addWeighted(img_bgr, 0.7, pred_color, 0.6, 0)
                
                mixed_color = cv2.add(gt_color, pred_color) 
                vis_mixed = cv2.addWeighted(img_bgr, 0.5, mixed_color, 0.8, 0)
                
                final_vis = np.hstack([vis_gt, vis_pred, vis_mixed])
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(final_vis, "1. Ground Truth (30px)", (10, 40), font, 0.8, (0, 0, 255), 3)
                cv2.putText(final_vis, "2. Prediction (PCA 30px)", (img_bgr.shape[1] + 10, 40), font, 0.8, (0, 255, 0), 3)
                cv2.putText(final_vis, "3. Overlay (Yellow = Hit!)", (img_bgr.shape[1]*2 + 10, 40), font, 0.8, (0, 255, 255), 3)
                
                save_path = os.path.join(save_dir, f"lane_30px_eval_{count+1}.jpg")
                cv2.imwrite(save_path, final_vis)
                count += 1

def get_palette_from_cats(categories):
    """從 Category CSV 中提取真實的 RGB 顏色，對齊推論程式的顏色定義"""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i, c in enumerate(categories):
        if i < 256:
            palette[i] = [c.r, c.g, c.b]
    return palette

def _resolve_label_positions(items, H, W, font_scale=0.4):
    """
    貪婪標籤位置解算：嘗試多個外側候選位置，選第一個不與已擺放標籤重疊的位置。
    items: [(x1, y1, x2, y2, text, color), ...]
    return: [(bx1, by1, bx2, by2, tx, ty, text, color), ...]
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 3
    placed = []   # 已確定的標籤矩形 [(bx1,by1,bx2,by2), ...]
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
            clamp_rect(x1,        y1 - lh),      # 上方，左對齊
            clamp_rect(x2 - lw,   y1 - lh),      # 上方，右對齊
            clamp_rect(x1,        y2),            # 下方，左對齊
            clamp_rect(x2,        y1),            # 右側，頂對齊
            clamp_rect(x1 - lw,   y1),            # 左側，頂對齊
            clamp_rect(x1,        y1 - lh * 2),  # 更上方
            clamp_rect(x2,        y2 - lh),      # 右側，底對齊
        ]

        chosen = next((r for r in candidates if not overlaps_any(r)), candidates[0])
        placed.append(chosen)
        bx1, by1, bx2, by2 = chosen
        results.append((bx1, by1, bx2, by2, bx1 + pad, by2 - pad, text, color))

    return results

def debug_det_task(name, model, dataloader, device, categories, task,
                   num_samples=5, save_dir="debug/ts", conf_thresh=0.3, nms_thresh=0.45):
    """
    TS / TL 偵測 debug 視覺化：
    藍色框 = GT，彩色框 = 預測（依類別上色）。
    完全對齊 inference_multitask.py 的解碼邏輯。
    """
    if dataloader is None: return
    print(f"\n[DEBUG] 正在隨機抽取 {num_samples} 張 {name.upper()} 進行偵測視覺化...")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    palette = get_palette_from_cats(categories) if categories else np.zeros((256, 3), dtype=np.uint8)

    total_batches = len(dataloader)
    selected = set(random.sample(range(total_batches), min(num_samples, total_batches)))

    mean_np = np.array([0.485, 0.456, 0.406])
    std_np  = np.array([0.229, 0.224, 0.225])

    count = 0
    autocast_enabled = (device.type == 'cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=autocast_enabled):
        for batch_idx, data in enumerate(dataloader):
            if batch_idx not in selected:
                continue

            imgs    = data['img'].to(device)
            gt_data = data.get('raw_gt')   # [B, max_boxes, 5]  (x1y1x2y2 cls), -1 = padding

            outputs = model(pixel_values=imgs, task=task)
            final_bboxes, final_scores, final_classes = decode_yolox_outputs(
                outputs['logits'], conf_thresh=conf_thresh, nms_thresh=nms_thresh
            )

            B = imgs.shape[0]
            imgs_np = imgs.cpu().numpy()   # [B, 3, H, W]

            for b in range(B):
                if count >= num_samples:
                    print(f"  -> 成功儲存 {num_samples} 張 {name.upper()} 對照圖至 `{save_dir}/`")
                    return

                # ── 1. 還原圖片（反 Normalize） ─────────────────────────
                img = imgs_np[b].transpose(1, 2, 0)       # [H, W, 3]
                img = (std_np * img + mean_np).clip(0, 1)
                img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                H, W = img_bgr.shape[:2]

                vis_gt   = img_bgr.copy()
                vis_pred = img_bgr.copy()

                # ── 2. 左圖：GT ─────────────────────────────────────────
                if gt_data is not None:
                    gts = gt_data[b].cpu().numpy()         # [max_boxes, 5]
                    gt_items = []
                    for box in gts:
                        if box[4] < 0: continue            # padding
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        cls_id = int(box[4])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W-1, x2), min(H-1, y2)
                        cls_name = categories[cls_id].name if categories and cls_id < len(categories) else str(cls_id)
                        c     = palette[cls_id] if cls_id < len(palette) else [0, 200, 0]
                        color = (int(c[2]), int(c[1]), int(c[0]))   # RGB → BGR
                        cv2.rectangle(vis_gt, (x1, y1), (x2, y2), color, 2)
                        gt_items.append((x1, y1, x2, y2, cls_name, color))
                    for bx1, by1, bx2, by2, tx, ty, text, col in _resolve_label_positions(gt_items, H, W):
                        cv2.rectangle(vis_gt, (bx1, by1), (bx2, by2), col, -1)
                        cv2.putText(vis_gt, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                # ── 3. 右圖：Prediction（類別顏色框）────────────────────
                p_boxes  = final_bboxes[b]
                p_scores = final_scores[b]
                p_clses  = final_classes[b]
                if p_boxes.numel() > 0:
                    p_boxes_np  = p_boxes.cpu().numpy()
                    p_scores_np = p_scores.cpu().numpy()
                    p_clses_np  = p_clses.cpu().numpy().astype(int)
                    pred_items = []
                    for i in range(len(p_scores_np)):
                        x1, y1 = int(p_boxes_np[i][0]), int(p_boxes_np[i][1])
                        x2, y2 = int(p_boxes_np[i][2]), int(p_boxes_np[i][3])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W-1, x2), min(H-1, y2)
                        cls_id  = p_clses_np[i]
                        c       = palette[cls_id] if cls_id < len(palette) else [0, 255, 0]
                        color   = (int(c[2]), int(c[1]), int(c[0]))   # RGB → BGR
                        cv2.rectangle(vis_pred, (x1, y1), (x2, y2), color, 2)
                        cls_name = categories[cls_id].name if categories and cls_id < len(categories) else str(cls_id)
                        label    = f"{cls_name} {p_scores_np[i]:.2f}"
                        pred_items.append((x1, y1, x2, y2, label, color))
                    for bx1, by1, bx2, by2, tx, ty, text, col in _resolve_label_positions(pred_items, H, W):
                        cv2.rectangle(vis_pred, (bx1, by1), (bx2, by2), col, -1)
                        cv2.putText(vis_pred, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

                # ── 4. 標題列 + 水平拼接 ───────────────────────────────
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(vis_gt,   "Ground Truth", (8, 32), font, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(vis_gt,   "Ground Truth", (8, 32), font, 0.9, (0, 180, 0),     1, cv2.LINE_AA)
                cv2.putText(vis_pred, "Prediction",   (8, 32), font, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(vis_pred, "Prediction",   (8, 32), font, 0.9, (0, 120, 255),   1, cv2.LINE_AA)

                final_vis = np.hstack([vis_gt, vis_pred])

                save_path = os.path.join(save_dir, f"{name}_debug_{count + 1:03d}.jpg")
                cv2.imwrite(save_path, final_vis)
                count += 1

    if count > 0:
        print(f"  -> 成功儲存 {count} 張 {name.upper()} 對照圖至 `{save_dir}/`")

def test_rm(model, dataloader, device, cats_rm, num_samples=5, save_dir="debug_rm_source"):
    if dataloader is None: return
    print(f"\n[DEBUG] 正在隨機抽取 {num_samples} 張道路標線(RM)進行高畫質視覺化與 Error Map 分析...")
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 隨機挑 batch index，迭代時跳過不需要的 batch，避免整個驗證集 OOM
    total_batches = len(dataloader)
    selected = set(random.sample(range(total_batches), min(num_samples, total_batches)))

    palette = get_palette_from_cats(cats_rm)

    count = 0
    autocast_enabled = (device.type == 'cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=autocast_enabled):
        for batch_idx, data in enumerate(dataloader):
            if batch_idx not in selected:
                continue
            imgs = data['img'].to(device)
            img_paths = data['img_path']
            if 'ann' not in data: continue
            
            gt_masks = data['ann'].to(device)
            if gt_masks.dim() == 4:
                gt_masks = gt_masks.squeeze(1)
                
            # 模型推論
            outputs = model(pixel_values=imgs, task='rm')
            logits = outputs.get('logits', outputs.get('mask_logits'))
            
            # 在特徵圖層級取 Argmax 得到類別 (節省 GPU 記憶體)
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)
            gts = gt_masks.cpu().numpy().astype(np.uint8)
            
            B = imgs.shape[0]
            for b in range(B):
                if count >= num_samples:
                    print(f" 成功儲存 {num_samples} 張精緻版 RM 對照圖至 `{save_dir}/` 目錄下！")
                    return
                
                # 1. 讀取高畫質原圖
                img_bgr = cv2.imread(img_paths[b])
                if img_bgr is None: continue
                orig_h, orig_w = img_bgr.shape[:2]
                
                # 2. 將預測與標籤放大回「原始圖片」尺寸 (使用 NEAREST 避免產生無效的小數點類別)
                pred_resized = cv2.resize(preds[b], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                pred_resized = rm_post_process(pred_resized, min_area=int(orig_h * orig_w * 0.0003))
                gt_resized = cv2.resize(gts[b], (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                
                # 3. 高速上色 (Numpy 索引映射) 並將 RGB 轉為 OpenCV 需要的 BGR
                pred_color_bgr = cv2.cvtColor(palette[pred_resized], cv2.COLOR_RGB2BGR)
                gt_color_bgr = cv2.cvtColor(palette[gt_resized], cv2.COLOR_RGB2BGR)
                
                # 4. 建立 Error Map (綠色=預測正確，紅色=預測錯誤或漏抓)
                error_map = np.zeros_like(img_bgr)
                valid_mask = (gt_resized != 255) & ((gt_resized != 0) | (pred_resized != 0))
                correct_mask = (gt_resized == pred_resized) & valid_mask
                wrong_mask = (gt_resized != pred_resized) & valid_mask & (gt_resized != 255)
                
                error_map[correct_mask] = [0, 255, 0] # 綠色
                error_map[wrong_mask] = [0, 0, 255]   # 紅色
                
                # 5. 準備 3 張獨立的畫布
                vis_gt = img_bgr.copy()
                vis_pred = img_bgr.copy()
                vis_err = img_bgr.copy()
                
                # 6. 完美半透明疊加 (對齊 inference_multitask 邏輯：只在「有標線的前景區域」才疊加)
                alpha = 0.6
                
                # GT Overlay
                gt_fg = gt_resized > 0
                if gt_fg.any():
                    vis_gt[gt_fg] = cv2.addWeighted(img_bgr[gt_fg], 1 - alpha, gt_color_bgr[gt_fg], alpha, 0)
                
                # Pred Overlay
                pred_fg = pred_resized > 0
                if pred_fg.any():
                    vis_pred[pred_fg] = cv2.addWeighted(img_bgr[pred_fg], 1 - alpha, pred_color_bgr[pred_fg], alpha, 0)
                
                # Error Overlay
                err_fg = correct_mask | wrong_mask
                if err_fg.any():
                    vis_err[err_fg] = cv2.addWeighted(img_bgr[err_fg], 0.3, error_map[err_fg], 0.7, 0)
                
                # 7. 影像水平拼接
                final_vis = np.hstack([vis_gt, vis_pred, vis_err])
                
                # 8. 加上標籤文字 (自動適應寬度，字體調小確保清晰)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(final_vis, "1. Ground Truth", (10, 40), font, 0.8, (0, 0, 255), 2)
                cv2.putText(final_vis, "2. Prediction", (orig_w + 10, 40), font, 0.8, (0, 255, 0), 2)
                cv2.putText(final_vis, "3. Error Map (Green=OK, Red=Wrong)", (orig_w*2 + 10, 40), font, 0.8, (0, 255, 255), 2)
                
                # 存檔 (檔名加上原本的圖片名稱以利追蹤)
                img_stem = os.path.basename(img_paths[b])
                save_path = os.path.join(save_dir, f"rm_eval_{count+1}_{img_stem}")
                cv2.imwrite(save_path, final_vis)
                count += 1

# =================================================================================
# 3. 偵測與號誌指標 (TS/TL)
# =================================================================================
class RelaxedHitMetric:
    def __init__(self, num_classes, conf_thresh=0.3, iou_thresh=0.1):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.results = {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in range(num_classes)}

    def update(self, preds_boxes, preds_clses, preds_scores, gt_boxes, gt_clses):
        mask = preds_scores >= self.conf_thresh
        preds_boxes = preds_boxes[mask]
        preds_clses = preds_clses[mask]
        preds_scores = preds_scores[mask]

        for c in range(self.num_classes):
            p_mask = (preds_clses == c)
            g_mask = (gt_clses == c)
            p_b, p_s = preds_boxes[p_mask], preds_scores[p_mask]
            g_b = gt_boxes[g_mask]

            if len(g_b) == 0:
                self.results[c]['FP'] += len(p_b)
                continue
            if len(p_b) == 0:
                self.results[c]['FN'] += len(g_b)
                continue

            sort_idx = np.argsort(p_s)[::-1]
            p_b = p_b[sort_idx]
            ious = compute_iou_np(p_b, g_b)
            tp, fp = 0, 0
            gt_matched = np.zeros(len(g_b))

            for i in range(len(p_b)):
                max_iou = -1
                max_idx = -1
                for j in range(len(g_b)):
                    if gt_matched[j]: continue
                    if ious[i, j] > max_iou:
                        max_iou = ious[i, j]
                        max_idx = j
                
                if max_iou >= self.iou_thresh:
                    tp += 1
                    gt_matched[max_idx] = 1
                else:
                    fp += 1

            fn = len(g_b) - np.sum(gt_matched)
            self.results[c]['TP'] += tp
            self.results[c]['FP'] += fp
            self.results[c]['FN'] += fn

    def compute(self):
        total_tp = sum(self.results[c]['TP'] for c in range(self.num_classes))
        total_fp = sum(self.results[c]['FP'] for c in range(self.num_classes))
        total_fn = sum(self.results[c]['FN'] for c in range(self.num_classes))
        eps = 1e-8
        precision = total_tp / (total_tp + total_fp + eps)
        recall = total_tp / (total_tp + total_fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return {'Precision': precision, 'Recall': recall, 'F1': f1}

def filter_by_min_area(boxes, clses, scores=None, min_area=100):
    if len(boxes) == 0:
        if scores is not None: return boxes, clses, scores
        return boxes, clses
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    mask = areas >= min_area
    if scores is not None:
        return boxes[mask], clses[mask], scores[mask]
    return boxes[mask], clses[mask]

def box_area(boxes):
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 0)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    return w * h

def compute_iou_np(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-8)

class NativeCOCOMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.preds = []
        self.targets = []
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)
        self.area_ranges = {
            'all': (0, float('inf')),
            'small': (0, 32 ** 2),
            'medium': (32 ** 2, 96 ** 2),
            'large': (96 ** 2, float('inf'))
        }

    def update(self, preds_boxes, preds_clses, preds_scores, gt_boxes, gt_clses, img_scale=1.0):
        self.preds.append({'boxes': preds_boxes, 'classes': preds_clses, 'scores': preds_scores})
        self.targets.append({'boxes': gt_boxes, 'classes': gt_clses, 'scale': img_scale})

    def eval_ap(self, class_id, iou_thresh, area_range):
        min_area, max_area = area_range
        all_tp, all_fp, all_scores = [], [], []
        total_gt_in_range = 0

        for p, t in zip(self.preds, self.targets):
            gt_mask = (t['classes'] == class_id)
            gt_boxes = t['boxes'][gt_mask]
            p_mask = (p['classes'] == class_id)
            p_boxes, p_scores = p['boxes'][p_mask], p['scores'][p_mask]

            if len(gt_boxes) > 0:
                gt_areas = box_area(gt_boxes)
                img_scale = t.get('scale', 1.0)
                gt_areas_orig = gt_areas / (img_scale ** 2)
                gt_in_range_mask = (gt_areas_orig >= min_area) & (gt_areas_orig < max_area)
                total_gt_in_range += np.sum(gt_in_range_mask)
            else:
                gt_in_range_mask = np.array([], dtype=bool)

            if len(p_boxes) == 0: continue

            sort_idx = np.argsort(p_scores)[::-1]
            p_boxes, p_scores = p_boxes[sort_idx], p_scores[sort_idx]
            p_areas = box_area(p_boxes)

            if len(gt_boxes) == 0:
                p_in_range_mask = (p_areas >= min_area) & (p_areas < max_area)
                all_fp.extend(np.ones(np.sum(p_in_range_mask)))
                all_tp.extend(np.zeros(np.sum(p_in_range_mask)))
                all_scores.extend(p_scores[p_in_range_mask])
                continue

            ious = compute_iou_np(p_boxes, gt_boxes)
            gt_matched = np.zeros(len(gt_boxes))

            for i in range(len(p_boxes)):
                max_iou, max_idx = -1, -1
                for j in range(len(gt_boxes)):
                    if gt_matched[j]: continue
                    if ious[i, j] > max_iou:
                        max_iou, max_idx = ious[i, j], j
                
                if max_iou >= iou_thresh:
                    gt_matched[max_idx] = 1
                    if gt_in_range_mask[max_idx]:
                        all_tp.append(1); all_fp.append(0); all_scores.append(p_scores[i])
                else:
                    if (p_areas[i] >= min_area) and (p_areas[i] < max_area):
                        all_tp.append(0); all_fp.append(1); all_scores.append(p_scores[i])

        if total_gt_in_range == 0: return math.nan
        if len(all_scores) == 0: return 0.0

        all_tp, all_fp, all_scores = np.array(all_tp), np.array(all_fp), np.array(all_scores)
        sort_idx = np.argsort(all_scores)[::-1]
        all_tp, all_fp = all_tp[sort_idx], all_fp[sort_idx]
        acc_tp, acc_fp = np.cumsum(all_tp), np.cumsum(all_fp)
        
        eps = np.finfo(np.float64).eps
        recalls = acc_tp / total_gt_in_range
        precisions = acc_tp / np.maximum(acc_tp + acc_fp, eps)

        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = 0.0 if np.sum(recalls >= t) == 0 else np.max(precisions[recalls >= t])
            ap += p / 101.0
        return ap

    def compute(self):
        results = {'mAP': [], 'mAP_50': [], 'mAP_75': [], 'mAP_s': [], 'mAP_m': [], 'mAP_l': []}
        for c in range(self.num_classes):
            ap_matrix = {name: [] for name in self.area_ranges.keys()}
            for area_name, rng in self.area_ranges.items():
                for t in self.iou_thresholds:
                    ap = self.eval_ap(c, t, rng)
                    if not math.isnan(ap): ap_matrix[area_name].append(ap)
            
            if ap_matrix['all']:
                results['mAP'].append(np.mean(ap_matrix['all']))
                results['mAP_50'].append(ap_matrix['all'][0]) 
                if len(ap_matrix['all']) > 5: results['mAP_75'].append(ap_matrix['all'][5])
            if ap_matrix['small']: results['mAP_s'].append(np.mean(ap_matrix['small']))
            if ap_matrix['medium']: results['mAP_m'].append(np.mean(ap_matrix['medium']))
            if ap_matrix['large']: results['mAP_l'].append(np.mean(ap_matrix['large']))

        final_res = {}
        for k, v in results.items():
            final_res[k] = np.mean(v) if len(v) > 0 else float('nan')
        return final_res

def validate_seg_task(name, validator, categories, mode_name="Source"):
    if validator is None: return None, None
    print(f"\n[{name.upper()}] Evaluating {mode_name} Validation Set (mIoU & Per-class IoU)...")
    loss, miou, acc, iou_list = validator.validate()
    print(f"  -> {mode_name} mIoU: {miou:.4f} | Loss: {loss:.4f}")
    cat_ious = {}
    for idx, cat in enumerate(categories):
        if idx < len(iou_list):
            cat_ious[cat.name] = iou_list[idx]
    return miou, cat_ious

def validate_det_task(name, model, dataloader, device, num_classes, task, mode_name="Source", metric_type="coco", nms_thresh=0.50):
    if dataloader is None: return None
    
    min_box_area = 100 

    if metric_type == "coco":
        print(f"\n[{name.upper()}] Evaluating {mode_name} Validation Set (COCO mAP)...")
        metric = NativeCOCOMetric(num_classes=num_classes)
        decode_conf = 0.05
    else:
        print(f"\n[{name.upper()}] Evaluating {mode_name} Validation Set (Hit Metric: IoU>=0.1, Conf>=0.3)...")
        metric = RelaxedHitMetric(num_classes=num_classes, conf_thresh=0.3, iou_thresh=0.1)
        decode_conf = 0.05 
        
    model.eval()
    autocast_enabled = (device.type == 'cuda')
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=autocast_enabled):
        for i, data in enumerate(tqdm(dataloader, desc=f"  {name.upper()} {mode_name}", dynamic_ncols=True)):
            imgs = data['img'].to(device)
            if 'raw_gt' not in data: continue
            gt_data = data['raw_gt'].to(device)
            input_scales = data.get('input_scale')  # [B] tensor, present for TT100K/S2TLD

            outputs = model(pixel_values=imgs, task=task)['logits']
            final_bboxes, final_scores, final_classes = decode_yolox_outputs(outputs, conf_thresh=decode_conf, nms_thresh=nms_thresh)

            for b in range(imgs.shape[0]):
                p_boxes = final_bboxes[b].cpu().numpy() if len(final_bboxes[b]) > 0 else np.empty((0, 4))
                p_scores = final_scores[b].cpu().numpy() if len(final_scores[b]) > 0 else np.empty((0,))
                p_clses = final_classes[b].cpu().numpy() if len(final_classes[b]) > 0 else np.empty((0,))

                g_boxes = gt_data[b][:, :4]
                g_clses = gt_data[b][:, 4]
                valid_mask = g_clses != -1
                g_boxes, g_clses = g_boxes[valid_mask].cpu().numpy(), g_clses[valid_mask].cpu().numpy()

                g_boxes, g_clses = filter_by_min_area(g_boxes, g_clses, min_area=min_box_area)
                p_boxes, p_clses, p_scores = filter_by_min_area(p_boxes, p_clses, p_scores, min_area=min_box_area)

                sc = input_scales[b].item() if input_scales is not None else 1.0
                metric.update(p_boxes, p_clses, p_scores, g_boxes, g_clses, img_scale=sc)

    res = metric.compute()
    if metric_type == "coco":
        print(f"  -> {mode_name} mAP: {res['mAP']:.4f} | mAP@0.5: {res['mAP_50']:.4f} | mAP_s: {res['mAP_s']:.4f}")
    else:
        print(f"  -> {mode_name} Precision: {res['Precision']:.4f} | Recall: {res['Recall']:.4f} | F1: {res['F1']:.4f}")
    return res

# =================================================================================
# 主程式
# =================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Task UDA All Tasks Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)")
    # [NEW] 加入 --task 參數，預設為 all
    parser.add_argument("--task", type=str, default="all", choices=["rm", "ll", "ts", "tl", "all"], help="Choose specific task to evaluate, or 'all' for all tasks.")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_task = args.task  # 取得使用者指定的任務

    print(f"Loading configuration from {args.config}...")
    cfg = MultiTaskTrainingConfig.load(args.config)

    cats_rm = Category.load(cfg.category_csv_rlmd)
    cats_ll = Category.load(cfg.category_csv_ll)
    cats_ts = Category.load(cfg.category_csv_ts) if getattr(cfg, 'category_csv_ts', None) and os.path.exists(cfg.category_csv_ts) else []
    cfg.ts_num_classes = len(cats_ts) if len(cats_ts) > 0 else getattr(cfg, 'ts_num_classes', 13)
    cats_tl = Category.load(cfg.category_csv_tl) if getattr(cfg, 'category_csv_tl', None) and os.path.exists(cfg.category_csv_tl) else []

    def get_val_transforms(use_bdd_color_fix=False):
        t_list = [transform.LoadImg()]
        if use_bdd_color_fix:
            t_list.append(LoadBDDColorLabelToID())
        else:
            t_list.append(transform.LoadAnn())
        t_list.extend([
            transform.ToTensor(),
            transform.Resize(cfg.image_scale),
            transform.Normalize(),
            transform.Check(),
            Cleanup()
        ])
        return t_list

    transforms_rm_val = get_val_transforms(use_bdd_color_fix=False)
    
    # [修復] LL 的 Transform 必須與訓練時保持一致，不要加入 LoadBDDColorLabelToID
    # 解析度對齊 transforms_builder._LL_SIZE = [360, 640]（BDD100K 0.5× 等比縮放）
    def get_ll_val_transforms():
        t_list = [transform.LoadImg()]
        t_list.append(transform.ToTensor())
        t_list.append(transform.Resize([360, 640]))
        t_list.append(transform.Normalize())
        t_list.append(Cleanup())
        return t_list
    transforms_ll_val = get_ll_val_transforms()
    
    TS_INPUT_SIZE = (960, 960)

    val_loader_kwargs = {'batch_size': cfg.val_batch_size, 'shuffle': False, 'num_workers': 1, 'pin_memory': cfg.pin_memory}
    
    print(f"Initializing Datasets for task(s): {eval_task.upper()} ...")
    
    # ── 根據指定的 Task 動態載入 Dataset，節省時間與記憶體 ──
    # RM 
    if eval_task in ['rm', 'all']:
        ds_rm_src_val = get_dataset(cfg.dataset_rlmd, cfg.source_val_images_rlmd, cfg.source_val_labels_rlmd, None, transforms_rm_val)
        dl_rm_src_val = DataLoader(ds_rm_src_val, **val_loader_kwargs) if ds_rm_src_val else None
        ds_rm_tgt_val = get_dataset(cfg.dataset_rlmd, cfg.target_val_images_rlmd, cfg.target_val_labels_rlmd, None, transforms_rm_val) if getattr(cfg, 'target_val_images_rlmd', None) else None
        dl_rm_tgt_val = DataLoader(ds_rm_tgt_val, **val_loader_kwargs) if ds_rm_tgt_val else None
    else:
        dl_rm_src_val = dl_rm_tgt_val = None

    # LL
    if eval_task in ['ll', 'all']:
        ds_ll_src_val = get_dataset(cfg.dataset_ll, cfg.source_val_images_ll, cfg.source_val_labels_ll, None, transforms_ll_val)
        dl_ll_src_val = DataLoader(ds_ll_src_val, **val_loader_kwargs) if ds_ll_src_val else None
        ds_ll_tgt_val = get_dataset(cfg.dataset_ll, cfg.target_val_images_ll, cfg.target_val_labels_ll, None, transforms_ll_val) if getattr(cfg, 'target_val_images_ll', None) else None
        dl_ll_tgt_val = DataLoader(ds_ll_tgt_val, **val_loader_kwargs) if ds_ll_tgt_val else None
    else:
        dl_ll_src_val = dl_ll_tgt_val = None

    # TS
    if eval_task in ['ts', 'all']:
        ds_ts_src_val = get_dataset(cfg.dataset_ts, cfg.source_val_images_ts, cfg.source_val_labels_ts, rcm=None, transforms=None, input_size=TS_INPUT_SIZE) if getattr(cfg, 'source_val_images_ts', None) else None
        dl_ts_src_val = DataLoader(ds_ts_src_val, **val_loader_kwargs) if ds_ts_src_val else None
        ds_ts_tgt_val = get_dataset(cfg.dataset_ts, cfg.target_val_images_ts, cfg.target_val_labels_ts, rcm=None, transforms=None, input_size=TS_INPUT_SIZE) if getattr(cfg, 'target_val_images_ts', None) else None
        dl_ts_tgt_val = DataLoader(ds_ts_tgt_val, **val_loader_kwargs) if ds_ts_tgt_val else None
    else:
        dl_ts_src_val = dl_ts_tgt_val = None

    # TL
    if eval_task in ['tl', 'all']:
        ds_tl_src_val = get_dataset(cfg.dataset_tl, cfg.source_val_images_tl, cfg.source_val_labels_tl, rcm=None, transforms=None, input_size=TS_INPUT_SIZE) if getattr(cfg, 'source_val_images_tl', None) else None
        dl_tl_src_val = DataLoader(ds_tl_src_val, **val_loader_kwargs) if ds_tl_src_val else None
        ds_tl_tgt_val = get_dataset(cfg.dataset_tl, cfg.target_val_images_tl, cfg.target_val_labels_tl, rcm=None, transforms=None, input_size=TS_INPUT_SIZE) if getattr(cfg, 'target_val_images_tl', None) else None
        dl_tl_tgt_val = DataLoader(ds_tl_tgt_val, **val_loader_kwargs) if ds_tl_tgt_val else None
    else:
        dl_tl_src_val = dl_tl_tgt_val = None

    print("\nBuilding Multi-Task Model...")
    config_rm = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_rm.num_labels = len(cats_rm)
    config_ll = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_ll.num_labels = len(cats_ll)
    config_ts = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_ts.num_labels = cfg.ts_num_classes
    config_tl = SegformerConfig.from_pretrained("nvidia/mit-b1")
    config_tl.num_labels = getattr(cfg, 'tl_num_classes', 4)

    model = get_mt_model(config_rm, config_ll, config_ts, config_tl)
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # 與 inference_multitask.py 保持一致：優先載入 model_state_dict（學生模型）
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'ema' in ckpt:
        print("  -> model_state_dict not found, falling back to EMA weights...")
        ema_data = ckpt['ema']
        state_dict = ema_data.get('ema_model', ema_data)
    else:
        state_dict = ckpt  # bare state dict

    new_state_dict = {}
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor):
            continue  # 跳過 'step'/'initted' 等非 Tensor entry（防呆）
        name = k.replace("module.", "")
        if "discriminator" in name:
            continue
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # 初始化 RM 需要的變數
    if eval_task in ['rm', 'all']:
        # cfg.ignore_index 是 List[List[int]]，[0] 取出 RM 的 ignore list
        # Validator 期待 List[int]；Metrics 的 ignore_ids 也是 List[int]
        _raw_ignore = getattr(cfg, 'ignore_index', [[255]])
        class_ignore_rm = _raw_ignore[0] if _raw_ignore else [255]
        # 確保 255 一定在 ignore list 裡（通用 ignore index）
        if 255 not in class_ignore_rm:
            class_ignore_rm = list(class_ignore_rm) + [255]

        model_wrapper_rm = TaskModelWrapper(model, task='rm')

        metric_rm_src = Metrics(num_categories=len(cats_rm), ignore_ids=class_ignore_rm, nan_to_num=0)
        val_rm_src = Validator(
            TqdmDataLoader(dl_rm_src_val, "  RM Source"),
            model_wrapper_rm, device, metric_rm_src,
            cfg.crop_size, cfg.stride, len(cats_rm), 'slide', class_ignore_rm
        ) if dl_rm_src_val else None

        metric_rm_tgt = Metrics(num_categories=len(cats_rm), ignore_ids=class_ignore_rm, nan_to_num=0)
        val_rm_tgt = Validator(
            TqdmDataLoader(dl_rm_tgt_val, "  RM Target"),
            model_wrapper_rm, device, metric_rm_tgt,
            cfg.crop_size, cfg.stride, len(cats_rm), 'slide', class_ignore_rm
        ) if dl_rm_tgt_val else None

    print("\n" + "="*60)
    print(f"Starting Evaluation Pipeline for: {eval_task.upper()}")
    print("="*60)

    # 預設結果為 None
    rm_s_miou, rm_s_ious, rm_t_miou, rm_t_ious = None, None, None, None
    ll_s_res, ll_t_res = None, None
    ts_s_res, ts_t_res = None, None
    tl_s_res, tl_t_res = None, None

    # 1. 道路標線 RM
    if eval_task in ['rm', 'all']:
        rm_s_miou, rm_s_ious = validate_seg_task("rm", val_rm_src, cats_rm, mode_name="Source (Clear Daytime)")
        rm_t_miou, rm_t_ious = validate_seg_task("rm", val_rm_tgt, cats_rm, mode_name="Target (Night/Rainy)")
        test_rm(model, dl_rm_src_val, device, cats_rm, num_samples=10, save_dir="debug/rm/source")
        test_rm(model, dl_rm_tgt_val, device, cats_rm, num_samples=10, save_dir="debug/rm/target")

    # 2. 車道線 LL
    if eval_task in ['ll', 'all']:
        ll_s_res = evaluate_ll_task("ll", model, dl_ll_src_val, device, mode_name="Source (Clear Daytime)")
        ll_t_res = evaluate_ll_task("ll", model, dl_ll_tgt_val, device, mode_name="Target (Night/Rainy)")
        test_lane(model, dl_ll_src_val, device, num_samples=10, save_dir="debug/ll/source")
        test_lane(model, dl_ll_tgt_val, device, num_samples=10, save_dir="debug/ll/target")

    # 3. 交通標誌 TS
    if eval_task in ['ts', 'all']:
        ts_s_res = validate_det_task("ts", model, dl_ts_src_val, device, cfg.ts_num_classes, task='ts', mode_name="Source (Clear Daytime)", metric_type="coco")
        ts_t_res = validate_det_task("ts", model, dl_ts_tgt_val, device, cfg.ts_num_classes, task='ts', mode_name="Target (Night/Rainy)", metric_type="coco")
        debug_det_task("ts", model, dl_ts_src_val, device, cats_ts, task='ts', num_samples=10, save_dir="debug/ts/source")
        debug_det_task("ts", model, dl_ts_tgt_val, device, cats_ts, task='ts', num_samples=10, save_dir="debug/ts/target")

    # 4. 交通號誌 TL
    if eval_task in ['tl', 'all']:
        tl_s_res = validate_det_task("tl", model, dl_tl_src_val, device, config_tl.num_labels, task='tl', mode_name="Source (Clear Daytime)", metric_type="relaxed", nms_thresh=0.35)
        tl_t_res = validate_det_task("tl", model, dl_tl_tgt_val, device, config_tl.num_labels, task='tl', mode_name="Target (Night/Rainy)", metric_type="relaxed", nms_thresh=0.35)
        debug_det_task("tl", model, dl_tl_src_val, device, cats_tl, task='tl', num_samples=10, save_dir="debug/tl/source", nms_thresh=0.35)
        debug_det_task("tl", model, dl_tl_tgt_val, device, cats_tl, task='tl', num_samples=10, save_dir="debug/tl/target", nms_thresh=0.35)

    def format_score(score):
        return f"{score:.4f}" if score is not None and score > -1.0 and not math.isnan(score) else "N/A   "

    print("\n\n" + "="*80)
    print(f"{f'FINAL EVALUATION SUMMARY ({eval_task.upper()})':^80}")
    print("="*80)
    
    if eval_task in ['rm', 'll', 'all']:
        print(f"\n[ 1. Semantic Segmentation & Lane Line ]")
        print("-" * 65)
        print(f"{'Task / Metric':<30} | {'Source':<12} | {'Target':<12}")
        print("-" * 65)
        
        if eval_task in ['rm', 'all']:
            print(f"{'[RM] Road Marking (mIoU)':<30} | {format_score(rm_s_miou):<12} | {format_score(rm_t_miou):<12}")
            if rm_s_ious or rm_t_ious:
                for cat in cats_rm:
                    s_val = rm_s_ious.get(cat.name, math.nan) if rm_s_ious else math.nan
                    t_val = rm_t_ious.get(cat.name, math.nan) if rm_t_ious else math.nan
                    print(f"  - {cat.name:<27} | {format_score(s_val):<12} | {format_score(t_val):<12}")
        
        if eval_task in ['ll', 'all']:
            print("-" * 65)
            print(f"{'[LL] Lane Line':<30} | {'':<12} | {'':<12}")
            print(f"  - Pixel IoU                    | {format_score(ll_s_res['Pixel_IoU'] if ll_s_res else None):<12} | {format_score(ll_t_res['Pixel_IoU'] if ll_t_res else None):<12}")
            print(f"  - Accuracy (incl. background)  | {format_score(ll_s_res['Accuracy'] if ll_s_res else None):<12} | {format_score(ll_t_res['Accuracy'] if ll_t_res else None):<12}")
            print(f"  - F1-Score (IoU >= 0.5)        | {format_score(ll_s_res['F1@0.5'] if ll_s_res else None):<12} | {format_score(ll_t_res['F1@0.5'] if ll_t_res else None):<12}")
            print(f"  - F1-Score (IoU >= 0.3)        | {format_score(ll_s_res['F1@0.3'] if ll_s_res else None):<12} | {format_score(ll_t_res['F1@0.3'] if ll_t_res else None):<12}")
            
    if eval_task in ['ts', 'tl', 'all']:
        print(f"\n[ 2. Object Detection ]")
        
        if eval_task in ['ts', 'all']:
            print("-" * 80)
            print(f"[TS] Traffic Sign (COCO mAP)")
            print(f"{'Mode':<25} | {'mAP':<6} | {'mAP@.5':<6} | {'mAP@.75':<7} | {'mAP_s':<6} | {'mAP_m':<6} | {'mAP_l':<6}")
            print("-" * 80)
            def print_ts_row(mode, res_dict):
                if res_dict is None:
                    print(f"{mode:<25} | {'N/A':<6} | {'N/A':<6} | {'N/A':<7} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6}")
                else:
                    print(f"{mode:<25} | {format_score(res_dict.get('mAP', math.nan)):<6} | {format_score(res_dict.get('mAP_50', math.nan)):<6} | {format_score(res_dict.get('mAP_75', math.nan)):<7} | {format_score(res_dict.get('mAP_s', math.nan)):<6} | {format_score(res_dict.get('mAP_m', math.nan)):<6} | {format_score(res_dict.get('mAP_l', math.nan)):<6}")

            print_ts_row("Source (Clear Daytime)", ts_s_res)
            print_ts_row("Target (Night/Rainy)", ts_t_res)
        
        if eval_task in ['tl', 'all']:
            print("-" * 80)
            print(f"[TL] Traffic Light (Relaxed Hit Metric: Conf>0.3, IoU>0.1)")
            print(f"{'Mode':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
            print("-" * 80)
            def print_tl_row(mode, res_dict):
                if res_dict is None:
                    print(f"{mode:<25} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
                else:
                    print(f"{mode:<25} | {format_score(res_dict['Precision']):<10} | {format_score(res_dict['Recall']):<10} | {format_score(res_dict['F1']):<10}")

            print_tl_row("Source (Clear Daytime)", tl_s_res)
            print_tl_row("Target (Night/Rainy)", tl_t_res)
            
    print("="*80)

if __name__ == "__main__":
    main()