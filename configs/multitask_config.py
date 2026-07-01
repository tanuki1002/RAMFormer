from typing import Tuple, Optional, List, Dict, Any
from simple_parsing import Serializable
from dataclasses import dataclass, field

@dataclass
class MultiTaskTrainingConfig(Serializable):
    # --- Task 1: Road Marking (RLMD) ---
    dataset_rlmd: str
    category_csv_rlmd: str
    rcs_path_rlmd: Optional[str]
    source_train_images_rlmd: str
    source_train_labels_rlmd: str
    target_train_images_rlmd: List[str]
    source_val_images_rlmd: str
    source_val_labels_rlmd: str
    target_val_images_rlmd: List[str]
    target_val_labels_rlmd: List[str]

    # --- Task 2: Lane Line (BDD100K) ---
    dataset_ll: str
    category_csv_ll: str
    rcs_path_ll: Optional[str]
    source_train_images_ll: str
    source_train_labels_ll: str
    target_train_images_ll: List[str]
    source_val_images_ll: str
    source_val_labels_ll: str
    target_val_images_ll: List[str]
    target_val_labels_ll: List[str]

    # --- Task 3: Traffic Sign (TT100K) ---
    dataset_ts: str
    category_csv_ts: str
    rcs_path_ts: Optional[str]
    source_train_images_ts: str
    source_train_labels_ts: str
    target_train_images_ts: str
    source_val_images_ts: str
    source_val_labels_ts: str
    target_val_images_ts: str
    target_val_labels_ts: str
    ts_num_classes: int

    # --- Task 4: Traffic Light (S2TLD) ---
    dataset_tl: str
    category_csv_tl: str
    rcs_path_tl: Optional[str]
    source_train_images_tl: str
    source_train_labels_tl: str
    target_train_images_tl: str
    source_val_images_tl: str
    source_val_labels_tl: str
    target_val_images_tl: str
    target_val_labels_tl: str
    tl_num_classes: int

    # --- General Training Params ---
    model: str
    train_batch_size: int
    val_batch_size: int
    ignore_index: List[List[int]] # [[rm_ignore], [da_ignore]], [ll_ignore]]
    backbone_lr: float
    head_lr: float
    head_lr_rm: Optional[float]
    weight_decay: float
    pretrain_path: Optional[str]
    max_iters: int
    train_interval: int
    val_interval: int
    ema_update_intervals: List[int]
    
    # --- Augmentation & Preprocessing ---
    rcs_temperature: Optional[float]
    max_intensity: Optional[float]
    contrast_stretch: Optional[List[str]]
    img_proc_params: Optional[list]
    image_scale: Tuple[int, int]
    crop_size: Tuple[int, int]
    stride: Optional[Tuple[int, int]]
    random_resize_ratio: Tuple[float, float]
    mix_num: Optional[int]
    num_masks: int

    # --- System ---
    seed: int
    num_workers: int
    pin_memory: bool
    autocast: bool
    
    # --- Multi-Task Specific ---
    task_weight: Dict[str, float] = field(default_factory=lambda: {"rm": 1.0, "ll": 1.0, "ts": 1.0, "tl": 1.0})
    alternate_training: bool = False
    alternate_interval: int = 1
    milestone_iters: List[int] = field(default_factory=list)