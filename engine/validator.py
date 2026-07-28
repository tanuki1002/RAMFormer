import math
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from engine.metric import Metrics

class Validator:
    def __init__(
        self,
        dataloader: DataLoader,
        model: torch.nn.Module,
        device: torch.device,
        metric: Metrics,
        crop_size: Tuple[int, int],
        stride: Tuple[int, int],
        num_classes: int,
        mode: str,
        ignore_index: List[int] = list(),
        task: Optional[str] = None,
        conf_thresh: Optional[float] = None
    ):
        self.dataloader = dataloader
        self.model = model
        self.device = device
        self.metric = metric
        self.crop_size = crop_size
        self.stride = stride
        self.num_classes = num_classes
        self.mode = mode
        self.ignore_index = ignore_index
        self.task = task
        self.conf_thresh = conf_thresh
        assert mode == 'slide' or mode == 'basic', 'mode must be \'slide\' or \'basic\'.'

    def validate(self):
        avg_loss = 0
        avg_miou = 0
        avg_acc = 0
        iou_list = None
        
        self.model.eval()

        for data in self.dataloader:
            imgs = data["imgs"] if "imgs" in data else [data["img"]]
            ann = data["ann"]
            imgs, ann = [im.to(self.device) for im in imgs], ann.to(self.device)
            
            with torch.no_grad():
                logits = self.slide_inference(images=imgs)
            
            if self.num_classes > 1:
                if self.conf_thresh is not None:
                    probs = torch.softmax(logits, dim=1)
                    max_prob, pred_idx = probs.max(dim=1)
                    pred_idx[(pred_idx > 0) & (max_prob < self.conf_thresh)] = 0
                    predicted = pred_idx
                else:
                    predicted = logits.argmax(dim=1)
            else:
                predicted = (torch.sigmoid(logits.squeeze(1)) > 0.5).long()
            
            loss = torch.tensor(0.0).to(self.device)
            if self.num_classes > 1:
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=255)
                loss = loss_fct(logits, ann)
            elif self.num_classes == 1:
                valid_mask = ((ann >= 0) & (ann != 255)).float()
                loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none")
                loss = loss_fct(logits.squeeze(1), ann.float())
                loss = (loss * valid_mask).mean()
            
            self.metric.compute_and_accum(predicted, ann)

            avg_loss += loss.item()

        iou = self.metric.get_and_reset()["IoU"]
        
        avg_loss = avg_loss / len(self.dataloader)
        valid_iou = [x for idx, x in enumerate(iou) if idx not in self.ignore_index]
        avg_miou = torch.Tensor(valid_iou).mean() if valid_iou else 0.0
        
        avg_acc = 0.0
        iou_list = iou
        for idx in range(len(iou_list)):
            if idx in self.ignore_index:
                iou_list[idx] = float('nan')

        return avg_loss, avg_miou, avg_acc, iou_list

    def frame_wise_validate(self):
        name_list = list()
        miou_list = list()
        
        self.model.eval()

        for data in self.dataloader:
            imgs = data["imgs"] if "imgs" in data else [data["img"]]
            ann = data["ann"]
            imgs, ann = [im.to(self.device) for im in imgs], ann.to(self.device)
            
            with torch.no_grad():
                logits = self.slide_inference(images=imgs)
            
            if self.conf_thresh is not None:
                probs = torch.softmax(logits, dim=1)
                max_prob, pred_idx = probs.max(dim=1)
                pred_idx[(pred_idx > 0) & (max_prob < self.conf_thresh)] = 0
                predicted = pred_idx
            else:
                predicted = logits.argmax(dim=1)

            self.metric.compute_and_accum(predicted, ann)
            iou = self.metric.get_and_reset()["IoU"]
            miou = 0.0
            cats = torch.unique(ann).tolist()
            valid_cats = 0
            for idx in range(len(iou)):
                if idx in cats and idx not in self.ignore_index:
                    miou += iou[idx]
                    valid_cats += 1
            if valid_cats > 0:
                miou /= valid_cats
            
            name_part = data["img_path"][0].split('/')[-1].split('.')[0]
            name_list.append(name_part)
            miou_list.append(miou)

        return name_list, miou_list
    
    def slide_inference(self, images: List[torch.Tensor]):
        inputs = list()
        h_stride, w_stride = self.stride
        h_crop, w_crop = self.crop_size
        batch_size, _, h_img, w_img = images[0].size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = images[0].new_zeros((batch_size, self.num_classes, h_img, w_img))
        count_mat = images[0].new_zeros((batch_size, 1, h_img, w_img))
        
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                
                inputs = [image[:, :, y1:y2, x1:x2] for image in images]
                
                forward_kwargs = {}
                if self.task is not None:
                    forward_kwargs['task'] = self.task

                all_logits = []
                for view_tensor in inputs:
                    out = self.model(pixel_values=view_tensor, **forward_kwargs)
                    if isinstance(out, dict):
                        all_logits.append(out['logits'])
                    elif isinstance(out, tuple):
                        all_logits.append(out[0])
                    else:
                        all_logits.append(out)
                upsampled_logits = torch.stack(all_logits).mean(0)

                preds += F.pad(
                    upsampled_logits,
                    (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
                )

                count_mat[:, :, y1:y2, x1:x2] += 1
        
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        return preds