# multi_task_segformer.py

from typing import Optional, Tuple, Union, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerDecodeHead, SegformerConfig
import math
from engine.lane_seg_decoder import LaneSegDecoder, ASPP
# ==========================================
# 1. Helper Modules
# ==========================================
class GradientScaler(torch.autograd.Function):
    """梯度縮放層：在前向傳播中保持不變，反向傳播時縮放梯度"""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def scale_gradient(x, scale):
    return GradientScaler.apply(x, scale)

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, 
                             padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        return -lambda_ * grad_input, None

def grad_reverse(x, lambd=1.0):
    lam = torch.tensor(lambd, device=x.device)
    return GradReverse.apply(x, lam)

# ==========================================
# [NEW] Geometry Adapter
# ==========================================
class GeometryAdapter(nn.Module):
    """
    幾何特徵適配器：結合 1x1 (通道) 和 3x3 (空間) Conv
    讓來自 Segmentation Backbone 的特徵能適應 Detection 任務
    """
    def __init__(self, in_channels, act="silu"):
        super().__init__()
        # 1. Channel-wise adaptation (線性轉換)
        self.channel_adapt = BaseConv(in_channels, in_channels, ksize=1, stride=1, act=act)
        
        # 2. Spatial awareness (輕量級深度可分離卷積)
        # 使用 Depthwise Conv 捕捉局部幾何特徵，參數量極少
        self.spatial_adapt = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1,
                      groups=in_channels, bias=False),  # Depthwise
            nn.BatchNorm2d(in_channels),
            nn.SiLU() if act == "silu" else nn.ReLU(inplace=True)
        )
        # Learnable scalar，初始化為 0.1
        # zeros(1) 會讓 spatial_adapt 分支在 early training 完全無作用；
        # 0.1 給予一個小的非零起點，讓空間幾何感知從第一個 iteration 就能貢獻梯度
        self.alpha = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x):
        x = self.channel_adapt(x)
        # 殘差連接：保留原始特徵，疊加空間特徵
        # x = x + self.spatial_adapt(x)
        # [修改] 使用 alpha 控制空間特徵的注入量
        x = x + self.alpha * self.spatial_adapt(x)  
        return x

# ==========================================
# [NEW] YOLOX Components (PAFPN + Decoupled Head)
# ==========================================
class SiLU(nn.Module):
    """Export-friendly SiLU activation"""
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

class BaseConv(nn.Module):
    """Standard Conv-BN-SiLU module"""
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = SiLU() if act == "silu" else (nn.ReLU(inplace=True) if act == "relu" else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YOLOXPAFPN(nn.Module):
    """
    Adapts SegFormer features to YOLOX PAFPN structure.
    Input: SegFormer features [P1, P2, P3, P4] -> We use [P2, P3, P4]
    """
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        # in_channels_list 對應 SegFormer 的 hidden_sizes (e.g., [64, 128, 320, 512])
        c3, c4, c5 = in_channels_list[1], in_channels_list[2], in_channels_list[3]

        # --- Top-down FPN Layers ---
        self.lateral_conv0 = BaseConv(c5, out_channels, 1, 1, act="silu") # P5 Input
        
        self.reduce_conv1 = BaseConv(c4, out_channels, 1, 1, act="silu") # P4 Input
        # [Fix] 輸入通道改為 out_channels * 2 (因為經過 Concat)
        self.C3_p4 = BaseConv(out_channels * 2, out_channels, 3, 1, act="silu") 

        self.reduce_conv2 = BaseConv(c3, out_channels, 1, 1, act="silu") # P3 Input
        # [Fix] 輸入通道改為 out_channels * 2
        self.C3_p3 = BaseConv(out_channels * 2, out_channels, 3, 1, act="silu")

        # --- Bottom-up PAN Layers ---
        self.bu_conv2 = BaseConv(out_channels, out_channels, 3, 2, act="silu") # P3->P4 Downsample
        # [Fix] 新增獨立的 Bottom-up Layer，輸入通道為 512
        self.bu_C3_p4 = BaseConv(out_channels * 2, out_channels, 3, 1, act="silu")
        
        self.bu_conv1 = BaseConv(out_channels, out_channels, 3, 2, act="silu") # P4->P5 Downsample
        # [Fix] 新增獨立的 Bottom-up Layer，輸入通道為 512
        self.bu_C3_p5 = BaseConv(out_channels * 2, out_channels, 3, 1, act="silu")

    def forward(self, input_features):
        # input_features: [P1, P2, P3, P4]
        x2, x1, x0 = input_features[1], input_features[2], input_features[3]

        # --- Top-down Path ---
        # Process P5 (x0)
        fpn_out0 = self.lateral_conv0(x0)  
        f_out0_upsample = F.interpolate(fpn_out0, scale_factor=2, mode="nearest")
        
        # Process P4 (x1) + Fuse with P5
        f_out1 = self.reduce_conv1(x1)
        f_out1 = torch.cat([f_out1, f_out0_upsample], 1) # Channel: 256+256=512
        f_out1 = self.C3_p4(f_out1) # 512 -> 256

        # Process P3 (x2) + Fuse with P4
        f_out1_upsample = F.interpolate(f_out1, scale_factor=2, mode="nearest")
        f_out2 = self.reduce_conv2(x2)
        f_out2 = torch.cat([f_out2, f_out1_upsample], 1) # Channel: 256+256=512
        f_out2 = self.C3_p3(f_out2) # 512 -> 256

        # --- Bottom-up Path ---
        p_out2 = f_out2  # P3 Output (Stride 8)
        
        # P3 -> P4
        bu_out1 = self.bu_conv2(p_out2) # Downsample
        bu_out1 = torch.cat([bu_out1, f_out1], 1) # Channel: 256+256=512
        p_out1 = self.bu_C3_p4(bu_out1) # 512 -> 256, P4 Output (Stride 16)
        
        # P4 -> P5
        bu_out0 = self.bu_conv1(p_out1) # Downsample
        bu_out0 = torch.cat([bu_out0, fpn_out0], 1) # Channel: 256+256=512
        p_out0 = self.bu_C3_p5(bu_out0) # 512 -> 256, P5 Output (Stride 32)

        return [p_out2, p_out1, p_out0]

class DecoupledHead(nn.Module):
    def __init__(self, num_classes, in_channels=256, act="silu"):
        super().__init__()
        self.num_classes = num_classes
        self.stem = BaseConv(in_channels, in_channels, 1, 1, act=act)
        
        # Classification branch
        self.cls_convs = nn.Sequential(
            BaseConv(in_channels, in_channels, 3, 1, act=act),
            BaseConv(in_channels, in_channels, 3, 1, act=act),
        )
        self.cls_preds = nn.Conv2d(in_channels, num_classes, 1, 1, 0)
        
        # Regression branch
        self.reg_convs = nn.Sequential(
            BaseConv(in_channels, in_channels, 3, 1, act=act),
            BaseConv(in_channels, in_channels, 3, 1, act=act),
        )
        self.reg_preds = nn.Conv2d(in_channels, 4, 1, 1, 0) # x, y, w, h
        self.obj_preds = nn.Conv2d(in_channels, 1, 1, 1, 0) # objectness

        # 初始化權重
        self.init_weights()
    
    def init_weights(self):
        # 針對 Objectness 分支做特殊初始化 (Prior Probability)
        # 讓初始預測機率約為 0.01 (背景)
        prior_prob = 1e-2
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        
        # 1. 通用初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 2. [優化] Regression 分支 - 更小的 std 防止初期發散
        for m in self.reg_preds.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001) 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 3. [優化] Classification 分支 - 稍大的 std 增強初期判別力
        nn.init.normal_(self.cls_preds.weight, std=0.015)

        # 4. Objectness - Prior Probability
        nn.init.constant_(self.obj_preds.bias, bias_value)

    def forward(self, x):
        x = self.stem(x)
        cls_feat = self.cls_convs(x)
        cls_output = self.cls_preds(cls_feat)
        reg_feat = self.reg_convs(x)
        reg_output = self.reg_preds(reg_feat)
        obj_output = self.obj_preds(reg_feat)
        return cls_output, reg_output, obj_output

class SegFormerYOLOXDecoder(nn.Module):
    def __init__(self, config: SegformerConfig, out_channels=256, gradient_scale=1.0):
        super().__init__()
        self.gradient_scale = gradient_scale
        # SegFormer B0-B5 hidden_sizes vary, e.g. [32, 64, 160, 256] for B0
        self.backbone_channels = config.hidden_sizes
        
        # [MODIFIED] 使用 Spatial Geometry Adapter
        # 對應 P2, P3, P4 (indices 1, 2, 3)
        self.det_adapters = nn.ModuleList([
            GeometryAdapter(self.backbone_channels[i], act="silu")
            for i in [1, 2, 3] 
        ])

        self.neck = YOLOXPAFPN(self.backbone_channels, out_channels)
        self.head = DecoupledHead(config.num_labels, out_channels)
        self.strides = [8, 16, 32]

    def forward(self, hidden_states, enable_gradient=True):
        # hidden_states: [P1, P2, P3, P4]
        
        # 1. 梯度縮放
        feats = [hidden_states[i] for i in [1, 2, 3]]
        
        if enable_gradient:
            feats = [scale_gradient(f, self.gradient_scale) for f in feats]
        else:
            feats = [f.detach() for f in feats]
            
        # 2. [新增] 通過 Geometry Adapter
        # 轉換語意特徵為幾何特徵，並增加空間感知
        adapted_feats = []
        for i, adapter in enumerate(self.det_adapters):
            adapted_feats.append(adapter(feats[i]))
            
        # 3. 構造 FPN 輸入
        # 我們的 YOLOXPAFPN 內部邏輯是取 input[1], input[2], input[3]
        # 所以這裡構造 [None, P2, P3, P4] 確保索引對齊
        fpn_input = [None, adapted_feats[0], adapted_feats[1], adapted_feats[2]]
  
        # 4. Feature Fusion (Neck)
        fpn_outs = self.neck(fpn_input)

        outputs = []
        for feat, stride in zip(fpn_outs, self.strides):
            cls_out, reg_out, obj_out = self.head(feat)
            outputs.append({
                'cls': cls_out,
                'reg': reg_out,
                'obj': obj_out,
                'stride': stride
            })
        return outputs

# ==========================================
# [NEW] 2D Sine-Cosine Positional Encoding
# ==========================================
class PositionEmbeddingSine2D(nn.Module):
    """
    輕量級的 2D 絕對位置編碼 (無須學習參數，完全依賴數學幾何)
    將 H 和 W 的座標轉換為 Sine 和 Cosine 波形，讓 Transformer 具備空間感知。
    """
    def __init__(self, hidden_dim=256, temperature=10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

    def forward(self, B, H, W, device):
        # 1. 建立 Y 和 X 的絕對座標網格
        y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).view(H, 1).expand(H, W)
        x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).view(1, W).expand(H, W)

        # 2. 將座標正規化到 [0, 2π] 之間
        eps = 1e-6
        y_embed = y_embed / (H + eps) * 2 * math.pi
        x_embed = x_embed / (W + eps) * 2 * math.pi

        # 3. 計算不同頻率的維度
        num_pos_feats = self.hidden_dim // 2
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)

        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t

        # 4. 交替使用 sin 和 cos
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(2)

        # 5. 組合 Y 和 X 的編碼
        pos = torch.cat((pos_y, pos_x), dim=2) 
        pos = pos.unsqueeze(0).expand(B, H, W, self.hidden_dim).permute(0, 3, 1, 2) 
        
        return pos

# ==========================================
# 4. Standard Modules (Seg) & Main Model
# ==========================================
class SegformerMLP(nn.Module):
    def __init__(self, config, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)
    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))

class TaskSpecificDecoder(SegformerDecodeHead):
    def __init__(self, config):
        super().__init__(config)
        self.linear_c = nn.ModuleList([SegformerMLP(config, c) for c in config.hidden_sizes])
        self.linear_fuse = nn.Conv2d(config.decoder_hidden_size * 4, config.decoder_hidden_size, 1, bias=False)
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, 1)

    def forward(self, encoder_hidden_states):
        # 明確 override forward()，不依賴 SegformerDecodeHead 的內部實作
        # 若 HuggingFace 更版後 SegformerDecodeHead.forward() 改為回傳 SegformerSemanticSegmenterOutput，
        # compute_domain_discrimination_loss_mt 的 F.interpolate(dis_pred, ...) 就會 crash。
        # 此 override 確保永遠回傳 torch.Tensor，與 library 版本無關。
        batch_size = encoder_hidden_states[-1].shape[0]
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            h, w = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1).reshape(batch_size, -1, h, w)
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].shape[2:],
                mode="bilinear", align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)
        x = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        x = self.activation(self.batch_norm(x))
        x = self.dropout(x)
        return self.classifier(x)

class RMDecoder(SegformerDecodeHead):
    """
    增強版 RM Decoder：
      - ASPP：在 P1 解析度上補充多尺度空間形狀特徵（r=6/12/18 + GAP）
      - Conv3×3 pre-classifier：讓分類頭有局部空間感受野（原本僅 1×1）
      - Boundary Head：輔助邊界監督，強迫 pre_cls 特徵在相鄰類別邊界處具高歧異度
    TaskSpecificDecoder 保持不動，仍供 TaskSpecificDiscriminator 使用。
    """
    def __init__(self, config):
        super().__init__(config)
        # All-MLP 多尺度特徵提取（與 TaskSpecificDecoder 相同）
        self.linear_c = nn.ModuleList([SegformerMLP(config, c) for c in config.hidden_sizes])
        self.linear_fuse = nn.Conv2d(config.decoder_hidden_size * 4, config.decoder_hidden_size, 1, bias=False)
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()
        # ASPP：P1 解析度的多尺度空間上下文，補充 All-MLP 已有的跨尺度語意
        # RM 用小感受野 (2,4,6)：避免相鄰路標特徵互相污染，解決物件合併問題
        # LL 的 ASPP 保持預設 (6,12,18)：車道線需要長距離上下文
        self.aspp = ASPP(config.decoder_hidden_size, dilations=(2, 6, 14))
        # Conv3×3 pre-classifier：3×3 對應原圖 ~12×12px，能看到局部幾何形狀
        self.pre_cls = nn.Sequential(
            nn.Conv2d(config.decoder_hidden_size, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # 主分類頭：25 類語意分割
        self.classifier = nn.Conv2d(128, config.num_labels, 1)
        # 輔助邊界頭：共用 pre_cls 特徵，輸出 binary 邊界圖
        self.boundary_head = nn.Conv2d(128, 1, 1)

    def forward(self, encoder_hidden_states):
        batch_size = encoder_hidden_states[-1].shape[0]
        # Step 1：All-MLP —— 各尺度投影至 decoder_hidden_size，upsample 至 P1 大小
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            h, w = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)                              # Linear projection
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1).reshape(batch_size, -1, h, w)
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].shape[2:],
                mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)
        # Step 2：融合所有尺度（P4→P1 順序 concat）
        x = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        x = self.activation(self.batch_norm(x))
        x = self.dropout(x)
        # Step 3：ASPP —— 在 P1 特徵圖上取多感受野空間特徵
        x = self.aspp(x)
        # Step 4：Conv3×3 pre-classifier
        feat = self.pre_cls(x)
        # Step 5：主輸出（1/4 scale，交由 forward() 做 bilinear upsample）
        logits = self.classifier(feat)
        # Step 6：輔助邊界輸出（1/4 scale，與 logits_aux 同解析度）
        boundary = self.boundary_head(feat)
        return logits, boundary


class TaskSpecificDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head = TaskSpecificDecoder(config)
    def forward(self, x):
        scaled = [scale_gradient(s, 0.3) for s in x]   # P4 主要朝 task 方向更新，discriminator 做微調
        return self.head([grad_reverse(s) for s in scaled])

# 專為物件偵測 (TS/TL) 設計的多尺度領域鑑別器
# 同時對齊 P2（texture/細節）、P3（中階）、P4（語意），
# 使 GRL 梯度能覆蓋雨天/夜間造成的低階與高階 domain shift
class DetectionDiscriminator(nn.Module):
    def __init__(self, in_channels_list, hidden=64, grad_scale=0.1):
        """
        in_channels_list: [p2_ch, p3_ch, p4_ch]，例如 MiT-B0 = [64, 160, 256]
        """
        super().__init__()
        self.grad_scale = grad_scale
        p2_ch, p3_ch, p4_ch = in_channels_list

        # 各尺度投影到相同 hidden dim，再於 P4 空間融合
        self.proj_p2 = nn.Sequential(
            nn.Conv2d(p2_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj_p3 = nn.Sequential(
            nn.Conv2d(p3_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj_p4 = nn.Sequential(
            nn.Conv2d(p4_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden * 2, hidden, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 1, 3, 1, 1),
        )

    def forward(self, hidden_states):
        # hidden_states: (P1, P2, P3, P4)，index 0-3
        p2 = scale_gradient(grad_reverse(hidden_states[1]), self.grad_scale)
        p3 = scale_gradient(grad_reverse(hidden_states[2]), self.grad_scale)
        p4 = scale_gradient(grad_reverse(hidden_states[3]), self.grad_scale)

        target_size = p4.shape[-2:]   # P4 空間尺寸作為融合基準
        f2 = F.interpolate(self.proj_p2(p2), size=target_size,
                           mode='bilinear', align_corners=False)
        f3 = F.interpolate(self.proj_p3(p3), size=target_size,
                           mode='bilinear', align_corners=False)
        f4 = self.proj_p4(p4)
        return self.net(torch.cat([f2, f3, f4], dim=1))


# 車道線偵測用鑑別器
class LaneDiscriminator(nn.Module):
    """
    車道線專用鑑別器。
    接收 LightFPN 輸出 (stride-4, 128ch)，而非 P4。
    梯度反轉後流回 P1~P3，正確對齊車道線任務實際使用的特徵空間。
    """
    def __init__(self, in_channels=128, grad_scale=0.05):
        super().__init__()
        self.grad_scale = grad_scale
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, fpn_feat):
        x = grad_reverse(fpn_feat)
        x = scale_gradient(x, self.grad_scale)
        return self.net(x)


class MultiTaskSegformer(nn.Module):
    def __init__(self, config_rm, config_ll, config_ts, config_tl, ts_gradient_scale=0.1, tl_gradient_scale=0.1):
        super().__init__()
        self.encoder = SegformerModel(config_rm)
        self.decoder_rm = RMDecoder(config_rm)
        self.decoder_ll = LaneSegDecoder(config_rm, hidden_dim=128)
        self.decoder_ts = SegFormerYOLOXDecoder(config_ts, out_channels=128, gradient_scale=ts_gradient_scale)
        self.decoder_tl = SegFormerYOLOXDecoder(config_tl, out_channels=128, gradient_scale=tl_gradient_scale)

        # Discriminators
        conf_disc = SegformerConfig(**config_rm.to_dict())
        conf_disc.num_labels = 1

        # RM, LL 維持原始像素級鑑別器
        self.discriminator_rm = TaskSpecificDiscriminator(conf_disc)
        #self.discriminator_ll = TaskSpecificDiscriminator(conf_disc)
        
        self.discriminator_ll = LaneDiscriminator(in_channels=128, grad_scale=0.05)

        # TS, TL 使用多尺度鑑別器：P2+P3+P4 聯合對齊
        # MiT-B1: hidden_sizes = [64, 128, 320, 512]，取 index 1~3
        det_disc_channels = config_rm.hidden_sizes[1:4]  # B1: [128, 320, 512]
        self.discriminator_ts = DetectionDiscriminator(in_channels_list=det_disc_channels, grad_scale=ts_gradient_scale)
        self.discriminator_tl = DetectionDiscriminator(in_channels_list=det_disc_channels, grad_scale=tl_gradient_scale)
        
        self.config_rm = config_rm
        self.config_ll = config_ll

    def forward_encoder(self, x):
        return self.encoder(x, output_hidden_states=True, return_dict=True).hidden_states

    def train(self, mode: bool = True):
        super().train(mode)
        # backbone BN 永遠保持 eval，凍結 running stats
        # 原因：4 個任務依序 forward 會用不同尺度圖片（TS 960×960，RM/LL 512×512）
        # 連續更新同一組 BN running stats 會造成統計值混雜，每個任務的 normalized distribution 都不穩定
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, pixel_values, task='rm', **kwargs):
        outputs = self.encoder(pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        
        res = {}
        if task in ['ts', 'tl']:
            decoder = self.decoder_ts if task == 'ts' else self.decoder_tl 
            res["logits"] = decoder(hidden_states)
        elif task == 'll':
            # 縮小 LL 流回 encoder 的梯度強度，避免 ASPP/SCNN 汙染 RM 所需的精細特徵
            p1 = hidden_states[0].detach()           # P1 完全隔離，LL 只讀不改
            p2 = scale_gradient(hidden_states[1], 0.05)
            p3 = scale_gradient(hidden_states[2], 0.05)
            out = self.decoder_ll((p1, p2, p3))
            mask_logits = F.interpolate(
                out['mask_logits'],
                size=pixel_values.shape[-2:],
                mode='bilinear', align_corners=False
            )
            res['mask_logits'] = mask_logits
        elif task == 'rm':
            logits_small, boundary_small = self.decoder_rm(hidden_states)  # [B,C,H/4,W/4], [B,1,H/4,W/4]
            logits = F.interpolate(logits_small, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
            res["logits"]          = logits
            res["logits_aux"]      = logits_small    # 供 deep supervision 使用
            res["boundary_logits"] = boundary_small  # 供 boundary loss 使用，1/4 scale
        
        # [NEW] 將 Encoder 抽取的特徵也一起回傳，供 Discriminator 重用
        res["hidden_states"] = hidden_states
        
        return res
    
    def forward_discriminator(self, hidden_states, task='rm'):
        """
        專門用於執行 Discriminator 的接口。
        輸入是 Encoder 提取出的 hidden_states。
        """
        if task == 'rm': return self.discriminator_rm(hidden_states)
        if task == 'll':
            p1 = hidden_states[0].detach()          # 與 forward() 的行為一致
            p2 = scale_gradient(hidden_states[1], 0.05)
            p3 = scale_gradient(hidden_states[2], 0.05)
            # fpn_feat 保留梯度圖：GRL 的 adversarial gradient 透過 fpn 傳回 p2/p3/encoder
            # fpn.parameters() 也會同時收到 adversarial gradient（domain-invariant 方向），與車道線
            # segmentation gradient（discriminative 方向）疊加 → 正確的域適應設計
            fpn_feat = self.decoder_ll.fpn(p1, p2, p3)
            return self.discriminator_ll(fpn_feat)
        if task == 'ts': return self.discriminator_ts(hidden_states)
        if task == 'tl': return self.discriminator_tl(hidden_states)
        return None

def get_model(config_rm, config_ll, config_ts, config_tl):
    # 預設 gradient_scale=0.1
    return MultiTaskSegformer(config_rm, config_ll, config_ts, config_tl, ts_gradient_scale=0.1, tl_gradient_scale=0.1)