"""
engine/decode_utils.py
────────────────────────────────────────────────────────────────────────
YOLOX 後處理工具：將模型多尺度輸出解碼成 BBox 並套用 NMS。
同時提供 Pseudo-label 格式化與物件感知遮罩工具，供 UDA target-domain 使用。

從 train_uda_multitask.py 抽出，原邏輯完全不變。
"""

import torch
import torchvision


def decode_yolox_outputs(
    outputs,
    K: int = 100,
    conf_thresh: float = 0.35,
    nms_thresh: float = 0.50,
):
    """
    將 SegFormerYOLOXDecoder 的多尺度輸出解碼成 BBox 列表。

    Args:
        outputs:      List[dict]，每個 dict 含 'cls', 'obj', 'reg', 'stride'
        conf_thresh:  objectness × class score 閾值
        nms_thresh:   NMS IoU 閾值

    Returns:
        final_bboxes  : List[Tensor]  每張圖的 BBox  [N, 4]  (x1,y1,x2,y2)
        final_scores  : List[Tensor]  每張圖的分數  [N]
        final_classes : List[Tensor]  每張圖的類別  [N]
    """
    final_bboxes, final_scores, final_classes = [], [], []
    batch_size = outputs[0]["cls"].shape[0]
    all_preds  = []

    for output in outputs:
        stride     = output["stride"]
        _, _, h, w = output["cls"].shape
        yv, xv     = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
        grid       = torch.stack((xv, yv), 2).view(1, -1, 2).to(output["cls"].device)

        cls_p = output["cls"].flatten(2).permute(0, 2, 1).sigmoid()
        obj_p = output["obj"].flatten(2).permute(0, 2, 1).sigmoid()
        reg_p = output["reg"].flatten(2).permute(0, 2, 1)

        # MODIFIED: No exp for w/h，與 YOLOXLoss.decode_box 保持一致
        cx     = (reg_p[..., 0] + grid[..., 0]) * stride
        cy     = (reg_p[..., 1] + grid[..., 1]) * stride
        ww     = (reg_p[..., 2] * stride).clamp(min=1e-3)
        hh     = (reg_p[..., 3] * stride).clamp(min=1e-3)
        bboxes = torch.stack([cx - ww / 2, cy - hh / 2, cx + ww / 2, cy + hh / 2], dim=-1)

        cls_val, cls_idx = torch.max(cls_p, dim=2)
        scores           = obj_p.squeeze(-1) * cls_val
        all_preds.append(
            torch.cat([bboxes, scores.unsqueeze(-1), cls_idx.float().unsqueeze(-1)], dim=-1)
        )

    all_preds = torch.cat(all_preds, dim=1)

    for b in range(batch_size):
        preds = all_preds[b]
        mask  = preds[:, 4] > conf_thresh
        preds = preds[mask]
        if len(preds) == 0:
            final_bboxes.append(torch.tensor([]))
            final_scores.append(torch.tensor([]))
            final_classes.append(torch.tensor([]))
            continue
        keep = torchvision.ops.nms(preds[:, :4], preds[:, 4], iou_threshold=nms_thresh)
        final_bboxes.append(preds[keep, :4])
        final_scores.append(preds[keep, 4])
        final_classes.append(preds[keep, 5])

    return final_bboxes, final_scores, final_classes