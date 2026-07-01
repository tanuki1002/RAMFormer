"""
engine/visualization.py
────────────────────────────────────────────────────────────────────────
TensorBoard 視覺化工具，供訓練中週期性 debug 使用。
從 train_uda_multitask.py 抽出，原邏輯完全不變。

Functions:
  visualize_validation_samples  — RM 語義分割預測色塊圖
  visualize_ts_predictions      — TS / TL BBox 預測疊加圖（通用）
  visualize_lstr_predictions    — LL 車道線 mask 疊加圖
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from engine.decode_utils import decode_yolox_outputs


def visualize_validation_samples(
    model,
    dataloader,
    palette,
    task: str,
    device,
    writer,
    iter_idx: int,
    num_samples: int = 3,
    title_prefix: str = "Val_Pred",
):
    """
    將語義分割預測結果寫入 TensorBoard。
    目前僅支援 'rm' 任務，傳入其他 task 時直接 return。
    """
    if task != "rm":
        return

    model.eval()
    try:
        if len(dataloader) == 0:
            return

        data         = next(iter(dataloader))
        imgs         = data["imgs"] if "imgs" in data else [data["img"]]
        input_tensor = imgs[0].to(device)
        batch_size   = input_tensor.shape[0]
        indices      = random.sample(range(batch_size), min(num_samples, batch_size))
        
        # 推論
        with torch.no_grad():
            outputs = model(pixel_values=input_tensor, task=task)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            preds = logits.argmax(dim=1)
        
        # 繪圖並寫入 TensorBoard
        for i, idx in enumerate(indices):
            pred_mask  = preds[idx].cpu().numpy().astype(np.uint8)
            color_mask = palette[pred_mask]
            tag        = f"{task.upper()}/{title_prefix}_{i}"
            writer.add_image(tag, color_mask, iter_idx, dataformats="HWC")

    except Exception as e:
        print(f"Visualization failed for task {task}: {e}")


def visualize_ts_predictions(
    model,
    dataloader,
    device,
    writer,
    iter_idx: int,
    num_samples: int = 3,
    confidence_threshold: float = 0.3,
    title: str = "TS_Pred_BBox",
    task: str = "ts",
):
    """
    將偵測任務 (TS / TL) 的 BBox 預測疊加到原圖，寫入 TensorBoard。
    TS 與 TL 共用此函式，透過 task 參數區分。
    """
    model.eval()
    try:
        if dataloader is None:
            return
        try:
            data = next(iter(dataloader))
        except StopIteration:
            return

        imgs       = data["img"].to(device)
        batch_size = imgs.shape[0]
        actual_n   = min(num_samples, batch_size)

        with torch.no_grad():
            multi_scale_outputs = model(pixel_values=imgs, task=task)["logits"]
            final_bboxes, final_scores, final_classes = decode_yolox_outputs(
                multi_scale_outputs, conf_thresh=0.05, nms_thresh=0.50
            )

        mean     = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        std      = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
        vis_imgs = torch.clamp((imgs * std + mean) * 255, 0, 255)

        for i in range(actual_n):
            img_np  = vis_imgs[i].permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()
            img_np  = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            bboxes  = final_bboxes[i]

            if len(bboxes) > 0:
                # 取得該張圖的預測
                bboxes  = bboxes.cpu().numpy()
                scores  = final_scores[i].cpu().numpy()
                classes = final_classes[i].cpu().numpy()
                for bbox, score, cls in zip(bboxes, scores, classes):
                    x1, y1, x2, y2 = bbox.astype(int)
                    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        img_np, f"{int(cls)}: {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )

            img_final = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            writer.add_image(
                f"{task.upper()}_Debug/{title}_{i}", img_final, iter_idx, dataformats="HWC"
            )

    except Exception as e:
        print(f"Visualization TS failed: {e}")


def visualize_lstr_predictions(
    model,
    dataloader,
    device,
    writer,
    iter_idx: int,
    num_samples: int = 3,
    title: str = "LL_Pred_Mask",
    task: str = "ll",
):
    """
    將車道線 mask 預測（綠色）與 GT（紅色）疊加到原圖，寫入 TensorBoard。
    """
    model.eval()
    try:
        if dataloader is None:
            return
        data     = next(iter(dataloader))
        imgs     = data["img"].to(device)
        actual_n = min(num_samples, imgs.shape[0])

        with torch.no_grad():
            out         = model(pixel_values=imgs, task=task)
            mask_logits = out.get("mask_logits", out.get("logits"))
            pred_masks  = (torch.sigmoid(mask_logits.squeeze(1)) > 0.5).cpu().numpy()

            if "lane_mask" in data:
                gt_mask = data["lane_mask"].to(device)
                if gt_mask.shape[-2:] != imgs.shape[-2:]:
                    gt_mask = F.interpolate(
                        gt_mask.unsqueeze(1).float(),
                        size=imgs.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                data["lane_mask"] = gt_mask

        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)
        vis  = torch.clamp((imgs * std + mean) * 255, 0, 255)

        for i in range(actual_n):
            img_np  = vis[i].permute(1, 2, 0).cpu().numpy().astype("uint8").copy()
            mask    = pred_masks[i].astype("uint8")
            overlay = img_np.copy()
            # 綠色 = 預測前景
            overlay[mask == 1] = (
                overlay[mask == 1] * 0.5 + np.array([0, 255, 0]) * 0.5
            ).astype("uint8")
            # 紅色 = GT
            if "lane_mask" in data:
                gt = data["lane_mask"][i].cpu().numpy().astype("uint8")
                overlay[gt == 1] = (
                    overlay[gt == 1] * 0.5 + np.array([255, 0, 0]) * 0.5
                ).astype("uint8")
            writer.add_image(f"LL_Debug/{title}_{i}", overlay, iter_idx, dataformats="HWC")

    except Exception as e:
        print(f"Visualization LL failed: {e}")