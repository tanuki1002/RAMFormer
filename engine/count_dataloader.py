import os
import cv2
import numpy as np
import warnings
import json
from PIL import Image
from typing import List, Any, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from engine.transform import Composition, Transform

class RLMDImgAnnDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        transforms: List[Transform],
    ) -> None:
        super().__init__()
        self.img_paths = list()
        self.ann_paths = list()
        for file in os.listdir(img_dir):
            filename = file[:-5] if file.endswith('.tiff') else file[:-4]
            if os.path.exists(f'{img_dir}/{file}') == True and os.path.exists(f'{ann_dir}/{filename}.png') == True:
                self.img_paths.append(f'{img_dir}/{file}')
                self.ann_paths.append(f'{ann_dir}/{filename}.png')
        self.transforms = Composition(transforms)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> List[Tuple[str, Any]]:
        return self.transforms.transform(
            {
                "img_path": str(self.img_paths[idx]),
                "ann_path": str(self.ann_paths[idx]),
            }
        )

class CityscapesImgAnnDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        ann_dir: str,
        transforms: List[Transform],
    ) -> None:
        super().__init__()
        self.img_paths = list()
        self.ann_paths = list()
        for folder in os.listdir(img_dir):
            for file in os.listdir(f'{img_dir}/{folder}'):
                if os.path.exists(f'{img_dir}/{folder}/{file}') == True and os.path.exists(f'{ann_dir}/{folder}/{file[:-17]}_gtFine_labelTrainIds.png') == True:
                    self.img_paths.append(f'{img_dir}/{folder}/{file}')
                    self.ann_paths.append(f'{ann_dir}/{folder}/{file[:-17]}_gtFine_labelTrainIds.png')
        self.transforms = Composition(transforms)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> List[Tuple[str, Any]]:
        return self.transforms.transform(
            {
                "img_path": str(self.img_paths[idx]),
                "ann_path": str(self.ann_paths[idx]),
            }
        )