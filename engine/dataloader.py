import os
import cv2
import numpy as np
import warnings
import json
import random
from PIL import Image, ImageFilter, ImageEnhance
from typing import List, Optional, Union
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from engine.transform import Composition, Transform
from engine.category import Category
from typing import Tuple
import glob
warnings.filterwarnings("ignore", category=UserWarning)

class RareCategoryManager:
    def __init__(
        self,
        categories: List[Category],
        rcs_path: str,
        temperature: float,
    ) -> None:
        with open(rcs_path) as f:
            data = json.load(f)

        self.stems = {cat.id: [] for cat in categories}
        self.category_probs = torch.zeros(len(categories))
        for d in data:
            count = np.array(d["count"])
            for cat in categories:
                if count[cat.id]:
                    self.stems[cat.id].append(d["filename"])
            self.category_probs += count
        self.consumable_stems = deepcopy(self.stems)
        self.category_probs /= self.category_probs.sum()

        self.reverse_cate_probs = (1.0 - self.category_probs)
        # Handle case where reverse sum is 0 (single class dataset?)
        rev_sum = self.reverse_cate_probs.sum()
        if rev_sum > 0:
            self.reverse_cate_probs /= rev_sum
        else:
             # Fallback to uniform
            self.reverse_cate_probs[:] = 1.0 / len(self.reverse_cate_probs)

        self.apply_temperature(temperature)

        self.length = len(data)

    def apply_temperature(self, temperature: float) -> None:
        self.sampling_probs = ((1 - self.category_probs) / temperature).exp()
        div = self.sampling_probs[self.category_probs != 0].sum()
        if div > 0:
            self.sampling_probs /= div
        self.sampling_probs[self.category_probs == 0] = 0

    def get_rare_cat_id(self) -> int:
        return np.random.choice(
            [i for i in range(len(self.sampling_probs))],
            replace=True,
            p=self.sampling_probs.numpy(),
        )
    
    def get_mix_cat_id(self, cateList: List[int], mix_num: int = 1) -> list:
        # [Fix] 確保 cateList 中的 ID 在合法範圍內
        valid_cats = [cat for cat in cateList if 0 <= cat < len(self.reverse_cate_probs)]
        
        if not valid_cats:
            return []

        temp_probs = np.array([self.reverse_cate_probs[cat] for cat in valid_cats])
        total_prob = temp_probs.sum()
        
        # [Fix] 如果總和為 0 (例如全是常見類別) 或 NaN，改用均勻分佈
        if total_prob <= 1e-6 or np.isnan(total_prob):
            temp_probs = np.ones(len(valid_cats)) / len(valid_cats)
        else:
            temp_probs /= total_prob
            
        return np.unique(np.random.choice(
            valid_cats,
            size=min(mix_num, len(valid_cats)),
            replace=True,
            p=temp_probs
        )).tolist()

    def get_stems(self, i: int) -> List[Path]:
        if len(self.consumable_stems[i]) == 0:
            self.consumable_stems[i] = deepcopy(self.stems[i])
        return self.consumable_stems[i]

class RLMD(Dataset):
    def __init__(
        self,
        img_dir: Union[List[str], str],
        ann_dir: Union[List[str], str],
        rcm: Optional[RareCategoryManager],
        transforms: List[Transform]
    ):
        if isinstance(img_dir, str):
            img_dir = [img_dir]
        if ann_dir is None:
            ann_dir = [None for _ in range(0, len(img_dir))]
        elif isinstance(ann_dir, str):
            ann_dir = [ann_dir]
        assert ann_dir is None or len(img_dir) == len(ann_dir), f"Inconsistent number of dataset paths."
        
        self.img_paths = list()
        self.ann_paths = list()

        if rcm == None:
            for idx in range(0, len(img_dir)):
                for file in os.listdir(img_dir[idx]):
                    if ann_dir[idx] is None:
                        self.img_paths.append(f'{img_dir[idx]}/{file}')
                        self.ann_paths.append(None)
                    else:
                        ann_path = f'{ann_dir[idx]}/{file.split(".")[0]}.png'
                        if not os.path.exists(ann_path):
                            continue
                        self.img_paths.append(f'{img_dir[idx]}/{file}')
                        self.ann_paths.append(ann_path)

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.rcm = rcm
        self.transforms = Composition(transforms)

    def __len__(self):
        if self.rcm == None:
            return len(self.img_paths)
        else:
            return self.rcm.length

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.rcm == None:
            img_path = self.img_paths[idx]
            ann_path = self.ann_paths[idx]
        else:
            random_cat_id = self.rcm.get_rare_cat_id()
            stems = self.rcm.get_stems(random_cat_id)
            stem = random.choice(stems)
            stems.remove(stem)
            extension = 'jpg' if self.img_dir[0].split("/")[-1] == 'images' else 'tiff'
            img_path = f'{self.img_dir[0]}/{stem.split("/")[-1].split(".")[0]}.{extension}'
            ann_path = stem

        domain = 1 if 'rainy' in img_path else 0

        return self.transforms.transform(
            {
                "img_path": img_path,
                "ann_path": ann_path,
                "domain": domain
            }
        )

def parse_lstr_to_row_anchor(ann_path, orig_w, orig_h, num_rows=72, num_cols=100, max_lanes=7, target_h=512):
    """
    從你轉換好的 LSTR JSON (a,b,c,d) 轉成 Row Anchor Tensor
    回傳: [max_lanes, num_rows] 的 LongTensor，數值為 0~99，100 代表「此行無車道」
    """
    no_lane_idx = num_cols
    # 預設全部填滿 100 (無車道)
    gt = torch.full((max_lanes, num_rows), no_lane_idx, dtype=torch.long)
    
    if ann_path is None or not os.path.exists(ann_path):
        return gt
    
    with open(ann_path, 'r') as f:
        lane_data = json.load(f)
    
    # [修復] Y 網格點必須基於模型真實看到的輸入尺寸 (target_h)
    row_ys = np.linspace(target_h * 0.3, target_h, num_rows)
    
    # 計算 Y 軸的縮放比例
    scale_y = target_h / orig_h
    
    valid_lane_idx = 0
    for lane in lane_data.get("lanes", []):
        if valid_lane_idx >= max_lanes: break
            
        coeffs = lane["coefficients"]
        a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
        
        # [修復] y_start/y_end 也要縮放到 target_h 空間來做上下界判斷
        y_start, y_end = lane["y_start"] * scale_y, lane["y_end"] * scale_y
        y_min, y_max = min(y_start, y_end), max(y_start, y_end)
        
        for row_i, y_val in enumerate(row_ys):
            if y_val < y_min or y_val > y_max:
                continue
            
            # [修復] 代入多項式前，必須將 Y 座標還原回「原圖尺度」，因為 a,b,c,d 是在原圖擬合的
            y_orig = y_val / scale_y
            x_val = a * (y_orig**3) + b * (y_orig**2) + c * y_orig + d
            
            # X 的比例不因 Resize 而改變，可以直接算出 Bin Index
            bin_idx = int((x_val / orig_w) * num_cols)
            
            if 0 <= bin_idx < num_cols:
                gt[valid_lane_idx, row_i] = bin_idx
                
        valid_lane_idx += 1
            
    return gt

# =================================================================================
# [NEW] Mask 即時轉 Row Anchor 轉換器 (形態學虛線修復版)
# =================================================================================
def generate_row_anchors_from_mask(mask_tensor, num_rows=72, num_cols=100, max_lanes=7):
    """
    從二值化的像素 Mask 中動態提取出車道線的幾何網格點。
    [修復] 加入垂直形態學膨脹，解決「虛線斷裂導致 ID 錯亂」的致命問題！
    """
    mask = mask_tensor.squeeze().numpy().astype(np.uint8)
    H, W = mask.shape
    
    # 預設全部填滿 100 (無車道)
    gt = np.full((max_lanes, num_rows), num_cols, dtype=np.int64)
    
    # ---------------------------------------------------------
    # 🌟 核心魔術：垂直膨脹 (Vertical Dilation)
    # 建立一個「高度很長、寬度很窄」的 Kernel (例如 60x5)
    # 這樣只會在垂直方向把虛線的空隙填滿，不會把左右相鄰的車道線黏在一起
    # ---------------------------------------------------------
    kernel = np.ones((60, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 改對「黏合後」的 mask 進行連通集分析，這樣一整條虛線就會拿到同一個 ID！
    num_labels, labels = cv2.connectedComponents(dilated_mask, connectivity=8)
    
    row_ys = np.linspace(H * 0.3, H - 1, num_rows, dtype=int)
    
    lane_idx = 0
    for label_id in range(1, num_labels):
        if lane_idx >= max_lanes:
            break
            
        # 注意：提取實際 X 座標時，我們必須看「原始的綠色 mask」，
        # labels 只是用來幫我們分群而已！
        lane_pixels = (labels == label_id) & (mask == 1)
        
        # 過濾掉太小的雜訊 
        if np.sum(lane_pixels) < 30: 
            continue
            
        # 在每個 Y 軸網格尋找對應的 X 座標
        has_valid_points = False
        for row_i, y_val in enumerate(row_ys):
            y_min = max(0, y_val - 3)
            y_max = min(H, y_val + 3)
            
            row_slice = lane_pixels[y_min:y_max, :]
            coords = np.where(row_slice)[1] 
            
            if len(coords) > 0:
                avg_x = np.mean(coords)
                bin_idx = int((avg_x / W) * num_cols)
                bin_idx = max(0, min(bin_idx, num_cols - 1))
                gt[lane_idx, row_i] = bin_idx
                has_valid_points = True
                
        if has_valid_points:
            lane_idx += 1
            
    return torch.tensor(gt, dtype=torch.long)

class BDD100K(Dataset):
    def __init__(
        self,
        img_dir: Union[List[str], str],
        ann_dir: Union[List[str], str], # 這裡現在要傳入 LSTR JSON 的資料夾
        rcm: Optional[RareCategoryManager],
        transforms: List[Transform],
        max_lanes: int = 7, # 配合你的 Query 數量
        input_size: Tuple[int, int] = (512, 512) # [NEW] 接收外部傳入的尺寸
    ):
        if isinstance(img_dir, str):
            img_dir = [img_dir]
        if ann_dir is None:
            ann_dir = [None for _ in range(0, len(img_dir))]
        elif isinstance(ann_dir, str):
            ann_dir = [ann_dir]
        
        self.img_paths = list()
        self.ann_paths = list()
        self.max_lanes = max_lanes # 設定最大車道數

        if rcm is None:
            for idx in range(0, len(img_dir)):
                if not os.path.exists(img_dir[idx]): continue
                for file in os.listdir(img_dir[idx]):
                    if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                    
                    img_name_no_ext = os.path.splitext(file)[0]
                    img_full_path = os.path.join(img_dir[idx], file)
                    self.img_paths.append(img_full_path)
                    
                    if ann_dir[idx] is not None:
                        # [Modified] 改為尋找對應的 .json 檔案
                        png_path = os.path.join(ann_dir[idx], f'{img_name_no_ext}.png')
                        if os.path.exists(png_path):
                            self.ann_paths.append(png_path)
                        else:
                            self.ann_paths.append(None)
                    else:
                        self.ann_paths.append(None)

        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.rcm = rcm
        self.transforms = Composition(transforms)
        self.target_h = input_size[0] # [NEW] 儲存目標高度供標籤生成使用
        self.target_w = input_size[1]

    def __len__(self):
        if self.rcm == None:
            return len(self.img_paths)
        else:
            return self.rcm.length
    def load_lane_mask(self, mask_png_path: str, orig_h: int, orig_w: int):
        """
        直接讀取已轉換好的 mask PNG，保持原始輸入尺寸。
        輸出 torch.Tensor [orig_h, orig_w], float32, 0=背景, 1=車道線。
        
        在 BDD100K.__getitem__ 裡使用：
        # 假設你已經用 img = Image.open(...) 取得了原圖，並算出 orig_w, orig_h = img.size
        lane_mask = load_lane_mask(mask_path, orig_h=orig_h, orig_w=orig_w)
        """
        # 情況 1：完全沒有這個標註檔，產生與原圖同大小的空白 Mask
        if mask_png_path is None or not os.path.exists(mask_png_path):
            return torch.zeros((orig_h, orig_w), dtype=torch.float32)

        mask = cv2.imread(mask_png_path, cv2.IMREAD_GRAYSCALE)
        
        # 情況 2：圖檔損毀或讀取失敗
        if mask is None:
            return torch.zeros((orig_h, orig_w), dtype=torch.float32)

        # 情況 3：防呆機制。確保讀取到的 mask 尺寸與對應的 RGB 原圖絕對一致
        # （雖然通常標籤跟原圖會一樣大，但有時資料集會有意外）
        if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # 255 → 1.0 (二值化)
        mask_tensor = (mask > 127).astype(np.float32)
        
        return torch.from_numpy(mask_tensor)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.rcm == None:
            img_path = self.img_paths[idx]
            ann_path = self.ann_paths[idx]
        else:
            # UDA Target Domain 邏輯 (Teacher 不需要 ann_path)
            random_cat_id = self.rcm.get_rare_cat_id()
            stems = self.rcm.get_stems(random_cat_id)
            stem = random.choice(stems)
            extension = 'jpg' if 'images' in self.img_dir[0] else 'png'
            label_name_no_ext = os.path.splitext(os.path.basename(stem))[0]
            img_path = f'{self.img_dir[0]}/{label_name_no_ext}.{extension}'
            
            # [修正這裡] 強制指定副檔名為 .json，防止被 RCS 的 .png 誤導
            if self.ann_dir[0] is not None:
                # [修正] 既然現在都改成讀取 Segmentation Mask，這裡必須是 .png
                ann_path = os.path.join(self.ann_dir[0], f'{label_name_no_ext}.png')
            else:
                ann_path = None

        domain = 1 if 'rainy' in img_path else 0
        
        # 1. 讀取影像以獲得原始尺寸
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size 
        
        # 2. 先讀取 PNG Mask
        lane_mask = self.load_lane_mask(ann_path, orig_h=orig_h, orig_w=orig_w)
        
        # 3. [核心修復] 嘗試讀 JSON，若沒有 JSON 就用 Mask 即時推算 Anchor！
        json_path = ann_path.replace('.png', '.json') if ann_path else None
        if json_path and os.path.exists(json_path):
            row_anchor_gt = parse_lstr_to_row_anchor(
                json_path, orig_w, orig_h, 
                num_rows=72, num_cols=100, max_lanes=7, target_h=self.target_h
            )
        else:
            # 呼叫我們剛寫好的救星函數
            row_anchor_gt = generate_row_anchors_from_mask(
                lane_mask, num_rows=72, num_cols=100, max_lanes=7
            )

        data_dict = {
            "img_path":  img_path,
            "img":       img,
            "lane_mask": lane_mask,           
            "row_anchor_gt": row_anchor_gt,   
            "orig_size": (orig_w, orig_h),
            "domain":    domain
        }

        # 3. 執行 Transforms (做資料增強與轉 Tensor)
        transformed_data = self.transforms.transform(data_dict)
        
        # 4. 確保輸出是 Tensor (把這段補上對 row_anchor_gt 的保護)
        if "lane_mask" not in transformed_data:
            transformed_data["lane_mask"] = lane_mask
            
        # [NEW] 確保 row_anchor_gt 不被資料增強洗掉
        if "row_anchor_gt" not in transformed_data:
            transformed_data["row_anchor_gt"] = row_anchor_gt

        return transformed_data
    
def get_dataset(
    dataset_name: str,
    img_dir: Union[List[str], str],
    ann_dir: Union[List[str], str],
    rcm: Optional[RareCategoryManager],
    transforms: List[Transform],
    input_size: Tuple[int, int] = (512, 512),
    is_train: bool = False,               # [NEW] 判斷是否為訓練集
    cropped_data_dir: str = None          # [NEW] 存放切割素材的資料夾 (通用於標誌與號誌)
):
    dataset = globals().get(dataset_name)
    assert dataset, f"There is no {dataset} dataset in dataloader.py!"
    #return dataset(img_dir, ann_dir, rcm, transforms)
    # [Fix] 針對 TT100K 傳入 input_size，其他 dataset (RLMD/BDD) 保持原樣
    if dataset_name == "TT100K":
        return dataset(img_dir, ann_dir, rcm, transforms, input_size=input_size, is_train=is_train, cropped_signs_dir=cropped_data_dir)
    elif dataset_name == "S2TLD":
        return dataset(img_dir, ann_dir, rcm, transforms, input_size=input_size, is_train=is_train, cropped_lights_dir=cropped_data_dir)
    elif dataset_name == "BDD100K": # [FIX] 確保 BDD100K 也收到正確的 input_size
        return dataset(img_dir, ann_dir, rcm, transforms, max_lanes=7, input_size=input_size)
    else:
        return dataset(img_dir, ann_dir, rcm, transforms)

class InfiniteDataloader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        drop_last: bool,
        pin_memory: bool,
    ) -> None:
        self.dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=None # Use default
        )
        self.iterator = iter(self.dataloader)

    def __iter__(self):
        return self.iterator

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)


# ==========================================
# TT100K Dataset
# ==========================================
class TT100K(Dataset):
    def __init__(self, 
        img_dir, 
        ann_dir, 
        rcm=None, 
        transforms=None, 
        input_size=(960, 960), 
        # output_stride 參數不再需要，因為我們不產出 Feature Map
        is_train=False,            # [NEW] 判斷是否為訓練集以啟用增強
        cropped_signs_dir=None     # [NEW] 存放切割好的交通標誌資料夾路徑
    ):
        self.img_dir = img_dir
        self.ann_dir = ann_dir 
        self.input_size = input_size
        self.transforms = transforms
        self.is_train = is_train
        
        # 獲取所有影像檔案名稱
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # 定義類別映射表 (請確保這裡與 Config 中的 num_classes 一致)
        self.cat_to_id = {
            "i4": 0, "i5": 1, "p11": 2, "p26": 3, "pl100": 4,
            "pl30": 5, "pl40": 6, "pl5": 7, "pl50": 8, "pl60": 9,
            "pl80": 10, "pn": 11, "pne": 12
        }
        self.num_classes = len(self.cat_to_id)
        # 設定最大物件數 (用於 Padding raw_gt 以便 batching)
        self.max_objs = 128

        # [NEW] 預先載入 Copy-Paste 需要的素材路徑
        self.cropped_signs = []
        if cropped_signs_dir and os.path.exists(cropped_signs_dir):
            # 假設資料夾結構為 cropped_signs_dir/類別名稱/圖片.jpg
            for cat_name in self.cat_to_id.keys():
                cat_path = os.path.join(cropped_signs_dir, cat_name)
                if os.path.exists(cat_path):
                    for img_file in glob.glob(os.path.join(cat_path, '*.*')):
                        self.cropped_signs.append({"class": cat_name, "path": img_file})
            print(f"[TT100K] 載入 {len(self.cropped_signs)} 張可用於 Copy-Paste 的標誌素材。")

    # 策略一：物件級別強增強 (模糊、過曝、過暗)
    def apply_instance_aug(self, img, bboxes):
        for obj in bboxes:
            if random.random() < 0.5: # 50% 機率對該物件進行增強
                x1, y1, x2, y2 = map(int, obj['bbox'])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 or y2 <= y1: continue
                
                # 挖出 BBox 內的圖像
                roi = img.crop((x1, y1, x2, y2))
                aug_type = random.choice(['blur', 'bright', 'dark'])
                
                # 模擬雨天/失焦
                if aug_type == 'blur':
                    roi = roi.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.5)))
                # 模擬逆光/強光
                elif aug_type == 'bright':
                    roi = ImageEnhance.Brightness(roi).enhance(random.uniform(1.5, 2.5))
                # 模擬夜晚/陰影
                else:
                    roi = ImageEnhance.Brightness(roi).enhance(random.uniform(0.3, 0.6))
                
                # 貼回原圖
                img.paste(roi, (x1, y1))
        return img

    # 策略二：Copy-Paste 拼貼
    def apply_copy_paste(self, img, bboxes):
        if not self.cropped_signs or random.random() < 0.30:   # 50% → 30% skip rate (70% 觸發)
            return img, bboxes

        num_pastes = random.randint(1, 4)   # 1~3 → 1~4
        pasted_boxes = []
        for _ in range(num_pastes):
            sign_info = random.choice(self.cropped_signs)
            try:
                sign_img = Image.open(sign_info['path']).convert('RGB')
            except:
                continue

            scale = random.uniform(0.3, 1.2)
            new_w, new_h = int(sign_img.width * scale), int(sign_img.height * scale)
            if new_w <= 4 or new_h <= 4: continue

            sign_img = sign_img.resize((new_w, new_h), Image.BILINEAR)
            max_x, max_y = img.width - new_w, img.height - new_h
            if max_x < 0 or max_y < 0: continue

            # 嘗試找不重疊的位置（最多試 10 次）
            placed = False
            for _ in range(10):
                px = random.randint(0, max_x)
                py = random.randint(0, max_y)
                new_box = [px, py, px + new_w, py + new_h]
                # 與已貼上的框計算 IoU，若都 < 0.3 才接受
                overlap = any(
                    self._box_iou(new_box, pb) > 0.3
                    for pb in pasted_boxes
                )
                if not overlap:
                    img.paste(sign_img, (px, py))
                    bboxes.append({"class": sign_info['class'], "bbox": new_box})
                    pasted_boxes.append(new_box)
                    placed = True
                    break

        return img, bboxes
    @staticmethod
    def _box_iou(b1, b2):
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)

        # 1. 讀取影像
        img = Image.open(img_path).convert('RGB')
        w_raw, h_raw = img.size

        # 2. 讀取標註 (加入 ann_dir 的保護機制)
        bboxes = []
        if self.ann_dir is not None:  # <--- [關鍵修復] 確保 Target Domain 不會去讀路徑
            ann_name = os.path.splitext(img_name)[0] + ".txt"
            ann_path = os.path.join(self.ann_dir, ann_name)

            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    for line in f.readlines():
                        items = line.strip().split()
                        if len(items) >= 5:
                            cls_name = items[0]
                            coords = [float(x) for x in items[1:]]
                            bboxes.append({"class": cls_name, "bbox": coords})

        # ==========================================
        # 執行數據增強
        # ==========================================
        if self.is_train:
            img = self.apply_instance_aug(img, bboxes)
            img, bboxes = self.apply_copy_paste(img, bboxes)

        # 3. Letterbox Resize (保持長寬比縮放)
        target_h, target_w = self.input_size
        scale = min(target_w / w_raw, target_h / h_raw)
        nw, nh = int(w_raw * scale), int(h_raw * scale)
        
        img_resized = img.resize((nw, nh), Image.BILINEAR)
        new_img = Image.new('RGB', (target_w, target_h), (128, 128, 128))
        new_img.paste(img_resized, (0, 0))
        
        img_np = np.ascontiguousarray(np.array(new_img).transpose(2, 0, 1))
        img_tensor = torch.tensor(img_np, dtype=torch.float32) / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # 4. 處理 Bounding Box (Raw GT)
        raw_gt = []
        for obj in bboxes:
            cls_id = self.cat_to_id.get(obj["class"])
            if cls_id is None: continue
            
            xmin, ymin, xmax, ymax = obj["bbox"]
            x1 = np.clip(xmin * scale, 0, target_w)
            y1 = np.clip(ymin * scale, 0, target_h)
            x2 = np.clip(xmax * scale, 0, target_w)
            y2 = np.clip(ymax * scale, 0, target_h)
            
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                raw_gt.append([x1, y1, x2, y2, float(cls_id)])

        # 5. Padding Raw GT
        raw_gt_np = np.full((self.max_objs, 5), -1.0, dtype=np.float32)
        if len(raw_gt) > 0:
            raw_gt_data = np.array(raw_gt, dtype=np.float32)
            num_objs = min(len(raw_gt), self.max_objs)
            raw_gt_np[:num_objs] = raw_gt_data[:num_objs]
            
        return {
            "img": img_tensor,
            "img_path": img_path,
            "raw_gt": torch.tensor(raw_gt_np),
            "input_scale": torch.tensor(scale, dtype=torch.float32),
        }

# ==========================================
# S2TLD Dataset (Traffic Light)
# ==========================================
class S2TLD(Dataset):
    def __init__(self, 
        img_dir, 
        ann_dir, 
        rcm=None, 
        transforms=None, 
        input_size=(960, 960),
        is_train=False,             # [NEW]
        cropped_lights_dir=None     # [NEW] 存放切割好的交通號誌資料夾 
    ):
        self.img_dir = img_dir
        self.ann_dir = ann_dir 
        self.input_size = input_size
        self.transforms = transforms
        self.is_train = is_train
        
        # 獲取所有影像檔案名稱
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.cat_to_id = {
            "green": 0, 
            "yellow": 1, 
            "red": 2, 
            "off": 3 
        }
        self.num_classes = len(self.cat_to_id)
        self.max_objs = 128

        # [NEW] 載入 Copy-Paste 素材
        self.cropped_lights = []
        if cropped_lights_dir and os.path.exists(cropped_lights_dir):
            for cat_name in self.cat_to_id.keys():
                cat_path = os.path.join(cropped_lights_dir, cat_name)
                if os.path.exists(cat_path):
                    for img_file in glob.glob(os.path.join(cat_path, '*.*')):
                        self.cropped_lights.append({"class": cat_name, "path": img_file})
            print(f"[S2TLD] 載入 {len(self.cropped_lights)} 張可用於 Copy-Paste 的號誌素材。")
    
    # 策略一：號誌特化增強 (模擬夜晚光暈、雨天模糊)
    def apply_instance_aug(self, img, bboxes):
        for obj in bboxes:
            if random.random() < 0.5:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 <= x1 or y2 <= y1: continue

                roi = img.crop((x1, y1, x2, y2))
                cls_name = obj.get('class', '')

                # off 燈號本身不發光，絕不套 glare（否則模型學到「高亮度 = off」的錯誤關聯）
                # off 只做模糊或去色，強化其「暗色、無彩度」的外觀特徵
                if cls_name == 'off':
                    aug_type = random.choice(['blur', 'dark', 'desaturate'])
                else:
                    aug_type = random.choice(['blur', 'glare', 'dark'])

                if aug_type == 'blur':
                    roi = roi.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.0)))
                elif aug_type == 'glare':
                    # 降低上限：2.0-3.0 會飽和成純白，喪失顏色資訊；1.5-2.2 保留色相
                    roi = ImageEnhance.Brightness(roi).enhance(random.uniform(1.5, 2.2))
                elif aug_type == 'desaturate':
                    # 去色：off 燈號沒有色彩飽和度，強化此特徵
                    roi = ImageEnhance.Color(roi).enhance(random.uniform(0.0, 0.3))
                else:
                    # 提高下限：0.2 太暗導致 off 與暗色背景無法區分
                    roi = ImageEnhance.Brightness(roi).enhance(random.uniform(0.35, 0.6))

                img.paste(roi, (x1, y1))
        return img

    # 策略二：Copy-Paste 拼貼
    def apply_copy_paste(self, img, bboxes):
        if not self.cropped_lights or random.random() < 0.35:
            return img, bboxes

        num_pastes = random.randint(1, 3)
        pasted_boxes = []
        for _ in range(num_pastes):
            light_info = random.choice(self.cropped_lights)
            try:
                light_img = Image.open(light_info['path']).convert('RGB')
            except:
                continue

            scale = random.uniform(0.3, 1.2)
            new_w, new_h = int(light_img.width * scale), int(light_img.height * scale)
            if new_w <= 4 or new_h <= 4: continue

            light_img = light_img.resize((new_w, new_h), Image.BILINEAR)
            max_x, max_y = img.width - new_w, img.height - new_h
            if max_x < 0 or max_y < 0: continue

            placed = False
            for _ in range(10):
                px = random.randint(0, max_x)
                py = random.randint(0, max_y)
                new_box = [px, py, px + new_w, py + new_h]
                overlap = any(
                    self._box_iou(new_box, pb) > 0.3
                    for pb in pasted_boxes
                )
                if not overlap:
                    img.paste(light_img, (px, py))
                    bboxes.append({"class": light_info['class'], "bbox": new_box})
                    pasted_boxes.append(new_box)
                    placed = True
                    break

        return img, bboxes
    @staticmethod
    def _box_iou(b1, b2):
        ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def apply_background_night_aug(self, img: Image.Image, bboxes: list) -> Image.Image:
        """只暗化背景，保留燈號區域原始外觀，讓模型學會在黑暗背景中辨識自發光的號誌燈。"""
        # 0.7 => 觸發機率 30%
        if random.random() > 0.7 or not bboxes:
            return img
        arr = np.array(img).astype(np.float32)
        H, W = arr.shape[:2]
        protect = np.zeros((H, W), dtype=np.float32)
        for obj in bboxes:
            x1, y1, x2, y2 = map(int, obj["bbox"])
            margin = 4
            x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(W, x2 + margin)
            y2m = min(H, y2 + margin)
            protect[y1m:y2m, x1m:x2m] = 1.0
        gamma  = random.uniform(1.5, 3.0)
        darken = np.power(arr / 255.0, gamma) * 255.0
        result = protect[:, :, None] * arr + (1 - protect[:, :, None]) * darken
        return Image.fromarray(result.clip(0, 255).astype(np.uint8))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.img_dir, img_name)

        # 1. 讀取影像
        img = Image.open(img_path).convert('RGB')
        w_raw, h_raw = img.size

        # 2. 讀取標註 (加入 ann_dir 的保護機制)
        bboxes = []
        if self.ann_dir is not None:  # <--- [關鍵修復] 確保 Target Domain 不會去讀路徑
            ann_name = os.path.splitext(img_name)[0] + ".txt"
            ann_path = os.path.join(self.ann_dir, ann_name)

            if os.path.exists(ann_path):
                with open(ann_path, 'r') as f:
                    for line in f.readlines():
                        items = line.strip().split()
                        if len(items) >= 5:
                            cls_name = items[0]
                            coords = [float(x) for x in items[1:]]
                            bboxes.append({"class": cls_name, "bbox": coords})

        # ==========================================
        # 執行數據增強
        # ==========================================
        if self.is_train:
            img = self.apply_background_night_aug(img, bboxes)
            img = self.apply_instance_aug(img, bboxes)
            img, bboxes = self.apply_copy_paste(img, bboxes)

        # 3. Letterbox Resize
        target_h, target_w = self.input_size
        scale = min(target_w / w_raw, target_h / h_raw)
        nw, nh = int(w_raw * scale), int(h_raw * scale)
        
        img_resized = img.resize((nw, nh), Image.BILINEAR)
        new_img = Image.new('RGB', (target_w, target_h), (128, 128, 128))
        new_img.paste(img_resized, (0, 0))
        
        img_np = np.ascontiguousarray(np.array(new_img).transpose(2, 0, 1))
        img_tensor = torch.tensor(img_np, dtype=torch.float32) / 255.0
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # 4. 處理 Bounding Box
        raw_gt = []
        for obj in bboxes:
            cls_id = self.cat_to_id.get(obj["class"])
            if cls_id is None: 
                try:
                    cls_id = int(obj["class"])
                except ValueError:
                    continue
            
            xmin, ymin, xmax, ymax = obj["bbox"]
            x1 = np.clip(xmin * scale, 0, target_w)
            y1 = np.clip(ymin * scale, 0, target_h)
            x2 = np.clip(xmax * scale, 0, target_w)
            y2 = np.clip(ymax * scale, 0, target_h)
            
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                raw_gt.append([x1, y1, x2, y2, float(cls_id)])

        # 5. Padding Raw GT
        raw_gt_np = np.full((self.max_objs, 5), -1.0, dtype=np.float32)
        if len(raw_gt) > 0:
            raw_gt_data = np.array(raw_gt, dtype=np.float32)
            num_objs = min(len(raw_gt), self.max_objs)
            raw_gt_np[:num_objs] = raw_gt_data[:num_objs]
            
        return {
            "img": img_tensor,
            "img_path": img_path,
            "raw_gt": torch.tensor(raw_gt_np),
            "input_scale": torch.tensor(scale, dtype=torch.float32),
        }
