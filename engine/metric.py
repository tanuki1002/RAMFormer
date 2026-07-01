# Most of the code pieces are adapted from MMSegmentation code base.
# 應該該是計算語義分割的評估指標 mIou 用
from collections import OrderedDict
from typing import Any, Optional, List, Dict, Tuple, Union
from numpy.typing import NDArray

import numpy as np
import torch


class Metrics:
    """
    Metrics computer by mmseg.

    num_categories (int): number of semantic categories.
    ignore_ids (int | list[int] | None): ignored categories when computing metrics.
    metric_types (list[MetricType]): metrics type that are going to be computed. Currently provided: iou, dice, fscore.
    """

    def __init__(
        self,
        num_categories: int,
        ignore_ids: Union[int, List[int]] = None,
        metric_types: List[str] = ["IoU", "Dice", "Fscore"],
        nan_to_num: float = None,
    ) -> None:
        self.num_classes = num_categories
        self.ignore_index = ignore_ids
        for t in metric_types:
            assert t in ["IoU", "Dice", "Fscore"]
        self.metric_types = metric_types
        self.nan_to_num = nan_to_num
        self.hist = 0

    def get_hist(self) -> torch.Tensor:
        """Get the accumulated histogram/confusion matrix.

        Returns:
            torch.Tensor: Accumulated histogram/confusion matrix
        """
        return self.hist

    def reset(self):
        self.hist = 0

    def compute_and_accum(self, pred: torch.Tensor, label: torch.Tensor) -> None:
        """Compute the histgram/confusion matrix and accumulate it.

        Args:
            pred (torch.Tensor): Prediction of a model. (NxWxH)
            label (torch.Tensor): Target of the prediction. (NxWxH)
        """
        # [Fix] 增加檢查，如果沒有數據傳入則不動作
        if pred.numel() == 0 or label.numel() == 0:
            return

        mask_hist = fast_hist(pred, label, self.num_classes, self.ignore_index)
        
        # 確保 self.hist 在正確的 device 上
        if isinstance(self.hist, int):
            self.hist = mask_hist
        else:
            self.hist += mask_hist.to(self.hist.device)

    def get_and_reset(self) -> Dict[str, NDArray]:
        """Get final result with the accumulated histogram/confusion matrix and reset.

        Returns:
            dict[str, NDArray]: Metric results.
        """
        # ==========================================
        # [Fix] 防呆機制：如果 self.hist 還是 int (沒有任何有效數據)，回傳 0 分
        # ==========================================
        if isinstance(self.hist, int):
            self.reset()
            return {
                "IoU": [0.0] * self.num_classes,
                "Accuracy": 0.0,
                "Mean IoU": 0.0
            }
        # ==========================================

        area_intersect = torch.diag(self.hist, 0)
        area_pred_label = torch.sum(self.hist, 0)
        area_label = torch.sum(self.hist, 1)
        area_union = area_pred_label + area_label - area_intersect

        self.hist = 0
        return total_area_to_metrics(
            area_intersect,
            area_union,
            area_pred_label,
            area_label,
            self.metric_types,
            self.nan_to_num,
        )


def total_area_to_metrics(
    total_area_intersect: np.ndarray,
    total_area_union: np.ndarray,
    total_area_pred_label: np.ndarray,
    total_area_label: np.ndarray,
    metric_types: List[str],
    nan_to_num: Optional[int] = None,
    beta: int = 1,
):
    """Calculate evaluation metrics
    Args:
        total_area_intersect (np.ndarray): The intersection of prediction
            and ground truth histogram on all classes.
        total_area_union (np.ndarray): The union of prediction and ground
            truth histogram on all classes.
        total_area_pred_label (np.ndarray): The prediction histogram on
            all classes.
        total_area_label (np.ndarray): The ground truth histogram on
            all classes.
        metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
            'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be
            replaced by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
    Returns:
        Dict[str, np.ndarray]: per category evaluation metrics,
            shape (num_classes, ).
    """

    def f_score(precision, recall, beta=1):
        """calculate the f-score value.

        Args:
            precision (float | torch.Tensor): The precision value.
            recall (float | torch.Tensor): The recall value.
            beta (int): Determines the weight of recall in the combined
                score. Default: 1.

        Returns:
            [torch.tensor]: The f-score value.
        """
        score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        return score

    # [Fix] 避免除以零
    total_label_sum = total_area_label.sum()
    if total_label_sum == 0:
        all_acc = 0.0
    else:
        all_acc = total_area_intersect.sum() / total_label_sum

    ret_metrics = OrderedDict({"aAcc": all_acc})
    for metric in metric_types:
        if metric == "IoU":
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics["IoU"] = iou
            ret_metrics["Acc"] = acc
        elif metric == "Dice":
            dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
            acc = total_area_intersect / total_area_label
            ret_metrics["Dice"] = dice
            ret_metrics["Acc"] = acc
        elif metric == "Fscore":
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)]
            )
            ret_metrics["Fscore"] = f_value
            ret_metrics["Precision"] = precision
            ret_metrics["Recall"] = recall

    ret_metrics = {metric: value.cpu().numpy() if torch.is_tensor(value) else value for metric, value in ret_metrics.items()}
    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {
                metric: np.nan_to_num(metric_value, nan=nan_to_num) if isinstance(metric_value, (np.ndarray, float, int)) else metric_value
                for metric, metric_value in ret_metrics.items()
            }
        )
    return ret_metrics


def fast_hist(
    pred: torch.Tensor,
    label: torch.Tensor,
    nc: int,
    ignore: Union[int, Tuple[int]] = None,
) -> torch.Tensor:
    """Compute class histogram.

    Args:
        pred (torch.Tensor): predicion index map, shape should be (BxHxW)
        label (torch.Tensor): label index map, shape should be (BxHxW)
        nc (int): number of classes.
        ignore (int, optional): Ignore index. Defaults to None.

    Returns:
        torch.Tensor: Class histgram to compute IoU.
    """
    pred = pred.flatten()
    label = label.flatten()

    if isinstance(ignore, int):
        mask = (pred >= 0) & (pred < nc) & (label != ignore)
    elif isinstance(ignore, list):
        mask = (pred >= 0) & (pred < nc)
        for index in ignore:
            mask = mask & (label != index)
    elif ignore is None:
        mask = (pred >= 0) & (pred < nc)
    else:
        raise TypeError("Ignore index should be int | list[int] | None.")

    # [Fix] 如果 mask 全是 False (所有 pixel 都被 ignore)，直接返回全 0 的矩陣
    if not mask.any():
        return torch.zeros((nc, nc), dtype=torch.float32, device=pred.device)

    return torch.bincount(nc * pred[mask] + label[mask], minlength=nc**2).reshape(
        nc, nc
    )