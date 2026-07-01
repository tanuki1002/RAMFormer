import sys
import os
import gc
import pandas as pd
import math
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from transformers import SegformerConfig

# [Modified] Import MultiTask modules
from engine.multi_task_segformer import get_model as get_mt_model, MultiTaskSegformer
from engine.dataloader import get_dataset
from engine.category import Category
from engine import transform
from engine.misc import set_seed
from engine.metric import Metrics
from engine.validator import Validator
from configs.config import EvaluationConfig

def main(cfg: EvaluationConfig, checkpoint: str):
    # 1. Load Categories
    categories = Category.load(cfg.category_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Setup Transforms
    # 注意：如果是 BDD 驗證集且有 Color Label，這裡可能需要像 train_uda_multitask 那樣用 LoadBDDColorLabelToID
    # 這裡暫時保持原樣，假設驗證集是乾淨的 ID Map
    val_transforms = [
        transform.LoadImg(),
        transform.LoadAnn(),
        *[transform.ContrastStretch(
                max_intensity=cfg.max_intensity,
                function_name=cfg.contrast_stretch[idx],
                parameter=cfg.img_proc_params[idx]
            ) for idx in range(0, len(cfg.contrast_stretch))
        ],
        transform.ToTensor(),
        transform.Resize(cfg.image_scale),
        transform.Normalize(),
    ]

    assert len(cfg.image_roots) == len(cfg.label_roots), f"Inconsistent number of dataset paths."

    # 3. Create Dataloaders
    dataloaders = [
        DataLoader(
            dataset=get_dataset(
                dataset_name=cfg.dataset, 
                img_dir=cfg.image_roots[idx],
                ann_dir=cfg.label_roots[idx],
                rcm=None,
                transforms=val_transforms
            ),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4, # 稍微提高 worker 數
            drop_last=False,
            pin_memory=cfg.pin_memory
        )
        for idx in range(0, len(cfg.image_roots))
    ]

    # 4. Metric
    metric = Metrics(num_categories=len(categories), ignore_ids=255, nan_to_num=0)

    # 5. [Modified] Model Initialization (MultiTask)
    # 由於我們需要載入多任務 Checkpoint，我們必須建立多任務模型結構
    # 這裡假設兩個任務用一樣的 config (或者你需要手動定義 rm/da 的 num_classes)
    
    # Config for RM
    segconfig_rm = SegformerConfig.from_pretrained("nvidia/mit-b1")
    segconfig_rm.num_labels = len(categories) # 假設 EvaluationConfig 的 csv 對應 RM

    segconfig_ll = SegformerConfig.from_pretrained("nvidia/mit-b1")
    segconfig_ll.num_labels = 2

    print("Building MultiTaskSegformer for evaluation...")
    model = get_mt_model(segconfig_rm, segconfig_ll)

    # 6. Load Checkpoint
    print(f"Loading checkpoint from {checkpoint}")
    ckpt = torch.load(checkpoint)
    
    # 處理可能的 key 不匹配 (例如 module. 前綴)
    state_dict = ckpt['model_state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") # remove 'module.'
        new_state_dict[name] = v
        
    try:
        model.load_state_dict(new_state_dict)
    except RuntimeError as e:
        print("Strict load failed, trying non-strict load (ignoring size mismatch in heads if any)...")
        print(e)
        model.load_state_dict(new_state_dict, strict=False)
        
    model.to(device)
    model.eval()

    # 7. [Modified] Setup Validators with Tasks
    # 根據你的設定檔，你需要決定哪個 dataloader 對應哪個 task
    # 這裡做一個簡單的假設：
    # 如果 dataset 名稱包含 'BDD'，設為 'da'，否則設為 'rm'
    # 或者你可以手動指定 list: tasks = ['rm', 'da']
    
    validators = []
    for idx in range(0, len(dataloaders)):
        # Heuristic to guess task, or generic default
        current_task = 'rm' # Default
        if 'lane' in cfg.dataset.lower() or 'll' in cfg.category_csv.lower():
            current_task = 'll'
        elif 'rlmd' in cfg.dataset.lower():
            current_task = 'rm'
            
        print(f"Dataset {idx} assigned to task: {current_task}")

        validators.append(
            Validator(
                dataloader=dataloaders[idx], 
                model=model, 
                device=device, 
                metric=metric, 
                crop_size=cfg.crop_size,
                stride=cfg.stride,
                num_classes=len(categories),
                mode='slide',
                ignore_index=cfg.ignore_index[idx],
                task=current_task # [New] Pass task
            )
        )

    # 8. Execution
    with torch.no_grad():
        for idx in range(0, len(validators)):
            loss, miou, _, iou_list = validators[idx].validate()
            print(f'Validation on Domain {idx} (Task: {validators[idx].task})')
            print(f"Loss: {loss}, Mean_iou: {miou}")
            print(pd.DataFrame({'Category': [cat.name for cat in categories], 'IoU': iou_list}))
            
            # Safe printing
            iou_strs = []
            for i in range(len(categories)):
                if i < len(iou_list):
                    val = iou_list[i]
                    iou_strs.append(f'{val * 100:.2f}' if not math.isnan(val) else 'NaN')
                else:
                    iou_strs.append('NaN')
            print(' & '.join(iou_strs))

if __name__ == "__main__":
    import sys

    assert len(sys.argv) == 3
    cfg = EvaluationConfig.load(sys.argv[1])
    checkpoint = sys.argv[2]
    main(cfg, checkpoint)