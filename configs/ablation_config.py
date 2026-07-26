from dataclasses import dataclass, field
from configs.multitask_config import MultiTaskTrainingConfig


@dataclass
class AblationConfig(MultiTaskTrainingConfig):
    """
    Extends MultiTaskTrainingConfig with per-component UDA flags for ablation studies.

    Ablation table (cumulative):
        Source-Only  : all False
        + FDA        : use_fda=True
        + Adversarial: use_fda=True, use_adversarial=True
        + EMA        : use_fda=True, use_adversarial=True, use_ema=True
        Full Model   : all True
    """
    # UDA component switches
    use_fda: bool = True
    use_adversarial: bool = True
    use_ema: bool = True
    use_classmix: bool = True

    # Task scheduling intervals (missing from base class, accessed via getattr in original)
    ll_interval: int = 2
    ts_interval: int = 2
    tl_interval: int = 4
