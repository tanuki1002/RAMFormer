# lane_seg_decoder.py  ── v5：YOLOP 啟發架構
# =====================================================================
# 相比 v4 的升級：
#   DilatedContextBlock (3 路) → ASPP (4 路 dilated + global avg pool)
#   感受野更大、更完整，直接對應 YOLOP 的 neck 設計理念
#   其餘（LightFPN P1+P2+P3、MessagePropagation、介面）保持不變
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig


# ──────────────────────────────────────────────────────────────────────
# DSConv: Depthwise Separable Conv + BN + ReLU
# ──────────────────────────────────────────────────────────────────────
class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, padding=pad,
                      dilation=dilation, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


# ──────────────────────────────────────────────────────────────────────
# LightFPN：P1 + P2 + P3 三路融合
#   P3 帶 RM backbone 習得的路面語意 → 抑制非路面假陽性
# ──────────────────────────────────────────────────────────────────────
class LightFPN(nn.Module):
    def __init__(self, p1_ch: int, p2_ch: int, p3_ch: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.proj_p1 = nn.Sequential(
            nn.Conv2d(p1_ch, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        self.proj_p2 = nn.Sequential(
            nn.Conv2d(p2_ch, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(p3_ch, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
        )

    def forward(self, p1, p2, p3):
        size = p1.shape[-2:]
        f1 = self.proj_p1(p1)
        f2 = F.interpolate(self.proj_p2(p2), size=size,
                           mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.proj_p3(p3), size=size,
                           mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([f1, f2, f3], dim=1))


# ──────────────────────────────────────────────────────────────────────
# ASPP (Atrous Spatial Pyramid Pooling)
# 參考 YOLOP / DeepLab 設計：
#   4 個不同膨脹率的並行 conv + 1 個全域平均池化分支
#   輸出通道 = hidden_dim（與輸入同）
# 相比舊的 DilatedContextBlock (r=1/6/12)，ASPP 多了 r=18 和 GAP：
#   r=18 → 覆蓋更遠的透視幾何（遠處車道線）
#   GAP  → 全域語意 → 幫助在光線不足 / 遮擋場景中推斷車道位置
# ──────────────────────────────────────────────────────────────────────
class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling，輸入輸出通道均為 channels。
    五個分支：1×1 conv, 3×3 r=d1, 3×3 r=d2, 3×3 r=d3, global avg pool。

    Args:
        channels: 輸入/輸出通道數
        dilations: 三個膨脹率的 tuple，預設 (6, 12, 18) 適合 LL（需長距離上下文）。
                   RM 任務建議用 (2, 4, 6)：縮小感受野，避免相鄰路標特徵互相污染。
    """
    def __init__(self, channels: int, dilations: tuple = (6, 12, 18)):
        super().__init__()
        c = channels // 5   # 每路輸出通道，最後拼接再投影回 channels

        self.b0 = nn.Sequential(              # 1×1 conv
            nn.Conv2d(channels, c, 1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.b1 = DSConv(channels, c, dilation=dilations[0])
        self.b2 = DSConv(channels, c, dilation=dilations[1])
        self.b3 = DSConv(channels, c, dilation=dilations[2])
        self.b4 = nn.Sequential(              # global average pool
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, c, 1, bias=False),
            nn.ReLU(inplace=True),             # 不用 BN：GAP 後空間為 1×1，BN 在 batch=1 時會 crash
        )
        self.proj = nn.Sequential(
            nn.Conv2d(c * 5, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        g = F.interpolate(self.b4(x), size=size,
                          mode='bilinear', align_corners=False)
        out = self.proj(torch.cat(
            [self.b0(x), self.b1(x), self.b2(x), self.b3(x), g], dim=1))
        return out + x   # 殘差連接


# ──────────────────────────────────────────────────────────────────────
# MessagePropagation：SCNN-style 四方向訊息傳播
#   沿行/列方向掃描 → 讓有車道的像素把訊息傳給虛線空隙
# ──────────────────────────────────────────────────────────────────────
class MessagePropagation(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        ph = kernel_size // 2
        pw = kernel_size // 2
        self.conv_r = nn.Sequential(
            nn.Conv2d(channels, channels, (1, kernel_size),
                      padding=(0, pw), bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.conv_l = nn.Sequential(
            nn.Conv2d(channels, channels, (1, kernel_size),
                      padding=(0, pw), bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.conv_d = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, 1),
                      padding=(ph, 0), bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.conv_u = nn.Sequential(
            nn.Conv2d(channels, channels, (kernel_size, 1),
                      padding=(ph, 0), bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fuse(torch.cat(
            [self.conv_r(x), self.conv_l(x),
             self.conv_d(x), self.conv_u(x)], dim=1))


# ──────────────────────────────────────────────────────────────────────
# LaneSegDecoder：主體
# ──────────────────────────────────────────────────────────────────────
class LaneSegDecoder(nn.Module):
    """
    Pipeline:
        [P1, P2, P3]  →  LightFPN  →  ASPP  →  MessagePropagation
                      →  seg_head  →  {'mask_logits': [B,1,H/4,W/4]}

    外部 (multi_task_segformer.py) 做 bilinear upsample 到原圖尺寸。
    HybridLaneDecoder 為向下相容別名。
    """
    def __init__(self, config: SegformerConfig, hidden_dim: int = 128):
        super().__init__()
        p1_ch = config.hidden_sizes[0]
        p2_ch = config.hidden_sizes[1]
        p3_ch = config.hidden_sizes[2]

        self.fpn  = LightFPN(p1_ch, p2_ch, p3_ch, hidden_dim)
        self.aspp = ASPP(hidden_dim)
        self.prop = MessagePropagation(hidden_dim, kernel_size=9)
        self.seg_head = nn.Sequential(
            DSConv(hidden_dim, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, hidden_states: tuple) -> dict:
        p1, p2, p3 = hidden_states[0], hidden_states[1], hidden_states[2]
        feat = self.fpn(p1, p2, p3)
        feat = self.aspp(feat)
        feat = self.prop(feat)
        return {'mask_logits': self.seg_head(feat), 'fpn_feat': feat}
    