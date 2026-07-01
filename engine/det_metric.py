# 給物件偵測用計算 mAP 的程式碼，這個腳本負責計算 IoU 並統計每個類別的 AP
# engine/det_metric.py
import numpy as np
import torch

def box_iou(box1, box2):
    """
    計算兩個 bbox 列表的 IoU
    box1: [N, 4] (x1, y1, x2, y2)
    box2: [M, 4] (x1, y1, x2, y2)
    Returns: [N, M]
    """
    # 確保是 numpy
    if isinstance(box1, torch.Tensor): box1 = box1.cpu().numpy()
    if isinstance(box2, torch.Tensor): box2 = box2.cpu().numpy()

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = np.maximum(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = np.minimum(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

class DetectionMetric:
    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        # 儲存所有的預測與 GT
        # preds[class_id] = list of [score, x1, y1, x2, y2]
        self.preds = {i: [] for i in range(self.num_classes)}
        # gts[class_id] = list of [x1, y1, x2, y2] for each image
        self.gts = {i: {} for i in range(self.num_classes)} 
        self.img_counter = 0

    def update(self, pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes):
        """
        更新一個 Batch 的資料
        pred_boxes: list of [x1, y1, x2, y2]
        gt_boxes: list of [x1, y1, x2, y2]
        """
        # 處理 Ground Truth
        # 為每張圖分配一個唯一 ID
        img_id = self.img_counter
        self.img_counter += 1

        # 整理 GT
        for cls, box in zip(gt_classes, gt_boxes):
            if int(cls) not in self.gts: self.gts[int(cls)] = {}
            if img_id not in self.gts[int(cls)]:
                self.gts[int(cls)][img_id] = []
            self.gts[int(cls)][img_id].append(box)

        # 整理 Prediction
        for cls, score, box in zip(pred_classes, pred_scores, pred_boxes):
            self.preds[int(cls)].append([float(score), box[0], box[1], box[2], box[3], img_id])

    def compute_map(self):
        aps = []
        for c in range(self.num_classes):
            if c not in self.gts or not self.preds[c]:
                # 如果該類別沒有 GT 或沒有預測，AP 為 0 (或者略過)
                # 這裡簡單處理：如果有 GT 但沒預測，AP=0；連 GT 都沒有，不計入平均
                if c in self.gts and self.gts[c]:
                    aps.append(0.0)
                continue

            # 1. 取出該類別所有預測並按分數排序
            preds = np.array(self.preds[c])
            if len(preds) == 0:
                aps.append(0.0)
                continue
                
            # sort by score descending
            sorted_ind = np.argsort(-preds[:, 0])
            preds = preds[sorted_ind]
            
            # 2. 計算 TP/FP
            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            
            # 記錄每張圖的 GT 是否被匹配過
            gt_matched = {} 
            # 統計該類別總共有多少個 GT
            n_pos = 0
            for img_id, boxes in self.gts[c].items():
                gt_matched[img_id] = np.zeros(len(boxes))
                n_pos += len(boxes)

            for i, p in enumerate(preds):
                pred_box = p[1:5]
                img_id = int(p[5])
                
                if img_id not in self.gts[c]:
                    fp[i] = 1
                    continue
                
                gt_boxes = np.array(self.gts[c][img_id])
                
                # 計算 IoU
                ious = box_iou(pred_box[None, :], gt_boxes)[0]
                
                if len(ious) > 0:
                    max_iou = np.max(ious)
                    max_idx = np.argmax(ious)
                    
                    if max_iou >= self.iou_threshold:
                        if gt_matched[img_id][max_idx] == 0:
                            tp[i] = 1
                            gt_matched[img_id][max_idx] = 1
                        else:
                            fp[i] = 1 # 已經被匹配過了 (Duplicate)
                    else:
                        fp[i] = 1
                else:
                    fp[i] = 1

            # 3. 計算 Precision / Recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recall = tp_cumsum / (n_pos + 1e-6)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

            # 4. 計算 AP (VOC 11-point 或 Area under curve)
            # 這裡使用簡單的 Area Under Curve
            ap = self.compute_ap_from_pr(precision, recall)
            aps.append(ap)

        return np.mean(aps) if aps else 0.0

    def compute_ap_from_pr(self, precision, recall):
        # 補齊頭尾以計算面積
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Integrate area under curve
        method = 'continuous' 
        if method == 'continuous':
            i = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        else:
            ap = 0 # Implement 11-point if needed
        return ap