# -*- coding: utf-8 -*-
"""
ESWA 公开模型：xin6.py 中 ResNet50 版 AdvancedProposedModel 的完整摘录。

与论文模块对应关系：
  - Preprocessing + AFFE (创新点1, Fig.6): PatchDCTFEM + 16/48 非对称划分 + 低频/高频卷积
  - FDAW            (创新点2, Fig.7): 频域差分正则 + 192 通道拼接 + 频域互补注意力
  - DAFM            (创新点3, Fig.8): 解码器-编码器 skip 上的差异感知特征调制
  - 空间编码器: ImageNet 预训练 ResNet50 的 conv1 / layer1 / layer2 / layer3
  - 解码器通道: 768 / 384 / 192 / 96，与 xin6.py DETERMINISTIC_MODE 下的 ResNet50 一致

权重键名与 xin6.AdvancedProposedModel 对齐，可直接加载 proposed_model / abl_resnet50 的 checkpoint。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

try:
    import torch_dct
except ImportError:
    torch_dct = None


def rgb_to_y(input_rgb):
    """RGB -> Y (ITU-R BT.601)，对应 Fig.6 的 Y Channel Extraction。"""
    r, g, b = input_rgb[:, 0:1], input_rgb[:, 1:2], input_rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


class ChannelAttention(nn.Module):
    """FDAW 末尾的 Frequency Complementary Attention（通道互补注意力）。"""

    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        actual_ratio = max(1, in_channels // 8)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, actual_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(actual_ratio, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class PatchDCTFEM(nn.Module):
    """
    Fig.6 上半部分 Preprocessing / Full-band Enhancement：
      Y 通道提取 -> 8x8 patch DCT -> base filter + 可学习掩膜
      -> 频带注意力 + 空间注意力 -> Fx (64-band)
    """

    def __init__(self, patch_size=8, high_band_threshold=16, embed_dim=64, num_heads=4):
        super().__init__()
        self.patch_size = patch_size
        self.high_band_threshold = high_band_threshold
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        base_mask = torch.zeros(64)
        base_mask[high_band_threshold:] = 1.0
        self.register_buffer("base_mask", base_mask.view(1, 64, 1, 1))
        self.learnable_mask = nn.Parameter(torch.zeros(1, 64, 1, 1))

        self.band_q = nn.Linear(embed_dim, embed_dim)
        self.band_k = nn.Linear(embed_dim, embed_dim)
        self.band_v = nn.Linear(embed_dim, embed_dim)
        self.band_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.space_q = nn.Linear(embed_dim, embed_dim)
        self.space_k = nn.Linear(embed_dim, embed_dim)
        self.space_v = nn.Linear(embed_dim, embed_dim)
        self.space_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

    def get_frequency_weight_map(self):
        with torch.no_grad():
            mask = torch.sigmoid(self.learnable_mask) + self.base_mask
            weight = (1.0 + mask).view(-1).detach().cpu().numpy()
            return weight.reshape(self.patch_size, self.patch_size)

    def forward(self, x):
        if torch_dct is None:
            raise ImportError("需要安装 torch_dct 才能使用 PatchDCTFEM")

        b, _, h, w = x.shape
        p = self.patch_size
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        y = rgb_to_y(x)
        patches = F.unfold(y, kernel_size=p, stride=p)
        n = patches.shape[-1]
        patches = patches.transpose(1, 2).contiguous().view(b * n, 1, p, p)
        dct_patch = torch_dct.dct_2d(patches, norm="ortho")
        dct_patch = dct_patch.reshape(b, n, p * p).transpose(1, 2).contiguous().view(
            b, 64, (h + pad_h) // p, (w + pad_w) // p
        )

        mask = torch.sigmoid(self.learnable_mask) + self.base_mask
        dct_patch = dct_patch * (1.0 + mask)

        _, c, h_, w_ = dct_patch.shape
        tokens_band = dct_patch.permute(0, 2, 3, 1).reshape(b * h_ * w_, c, 1).transpose(1, 2)
        band_out, _ = self.band_attn(tokens_band, tokens_band, tokens_band)
        band_out = band_out.transpose(1, 2).reshape(b, 64, h_, w_)

        tokens_space = dct_patch.reshape(b, 64, h_ * w_).transpose(1, 2)
        space_out, _ = self.space_attn(tokens_space, tokens_space, tokens_space)
        space_out = space_out.transpose(1, 2).reshape(b, 64, h_, w_)

        enhanced = dct_patch + 0.5 * band_out + 0.5 * space_out
        enhanced = F.interpolate(enhanced, size=(h + pad_h, w + pad_w), mode="bilinear", align_corners=True)
        enhanced = enhanced[:, :, :h, :w]
        return self.out_proj(enhanced)


class AsymmetricFrequencyFeatureExtraction(nn.Module):
    """
    AFFE / 创新点1 (Fig.6 下半部分):
      Fx(64-band) 上采样后按 16/48 非对称划分
      Flow : 16 -> 32 -> 64
      Fhigh: 48 -> 64 -> 128
    """

    def __init__(self):
        super().__init__()
        self.fem = PatchDCTFEM(patch_size=8, high_band_threshold=16, embed_dim=64, num_heads=4)
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        fem_feat = self.fem(x)
        low_map = fem_feat[:, :16, :, :]
        high_map = fem_feat[:, 16:, :, :]
        size = x.shape[2:]
        low_freq_feat = self.low_freq_conv(F.interpolate(low_map, size=size, mode="bilinear", align_corners=True))
        high_freq_feat = self.high_freq_conv(F.interpolate(high_map, size=size, mode="bilinear", align_corners=True))
        return low_freq_feat, high_freq_feat


class FrequencyDomainAdaptiveWeightedRegularization(nn.Module):
    """
    FDAW / 创新点2 (Fig.7):
      Flow2 --1x1--> 128
      Fhigh2 与低频投影做 Frequency Difference，再 128->64->128 得到 Fhigh3
      Concatenate(64+128=192) -> fusion -> Frequency Complementary Attention -> Ffreq
    """

    def __init__(self):
        super().__init__()
        self.low_freq_proj = nn.Conv2d(64, 128, kernel_size=1)
        self.difference_regularization = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(64 + 128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.freq_attention = ChannelAttention(128)

    def forward(self, low_freq_feat, high_freq_feat):
        low_freq_proj = self.low_freq_proj(low_freq_feat)
        freq_difference = high_freq_feat - low_freq_proj
        regularized_high_freq = self.difference_regularization(freq_difference)
        high_freq_feat = high_freq_feat + 0.3 * regularized_high_freq
        freq_feat = self.freq_fusion(torch.cat([low_freq_feat, high_freq_feat], dim=1))
        freq_feat = self.freq_attention(freq_feat) * freq_feat
        aux = {
            "lfrl": torch.mean(torch.abs(low_freq_feat)),
            "hfrl": torch.mean(torch.abs(regularized_high_freq)),
        }
        return freq_feat, high_freq_feat, aux


class FeatureAlignment(nn.Module):
    """空间-频域特征对齐（整体架构图中的 CF）。"""

    def __init__(self, in_rgb, in_freq, out_channels):
        super().__init__()
        self.rgb_proj = nn.Conv2d(in_rgb, out_channels, kernel_size=1)
        self.freq_proj = nn.Conv2d(in_freq, out_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat, freq_feat):
        freq_feat = F.interpolate(freq_feat, size=rgb_feat.shape[2:], mode="bilinear", align_corners=True)
        rgb_aligned = self.rgb_proj(rgb_feat)
        freq_aligned = self.freq_proj(freq_feat)
        gate = self.gate(rgb_aligned + freq_aligned)
        return rgb_aligned + gate * freq_aligned


class HORModule(nn.Module):
    """High-Order Relation：位置注意力 + 通道门控。"""

    def __init__(self, channels, reduction=16, num_heads=4):
        super().__init__()
        self.pos_attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(4, channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(4, channels // reduction), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.view(b, c, h * w).transpose(1, 2)
        attn_out, _ = self.pos_attn(tokens, tokens, tokens)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        return attn_out * self.channel_gate(attn_out)


class ComplementarityGuidedFeatureSelection(nn.Module):
    """
    DAFM / 创新点3 (Fig.8):
      Channel-Aligned Projection -> Difference Feature
      -> Channel / Spatial Complementarity Analysis
      -> Dual Complementarity Fusion -> Feature Adaptive Enhancement
    """

    def __init__(self, decoder_channels, encoder_channels, reduction=4):
        super().__init__()
        self.decoder_proj = nn.Conv2d(decoder_channels, encoder_channels, 1)
        self.encoder_proj = nn.Conv2d(encoder_channels, encoder_channels, 1)
        mid_channels = max(encoder_channels // reduction, 8)
        self.spatial_complement = nn.Sequential(
            nn.Conv2d(encoder_channels * 2, mid_channels, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
            nn.Sigmoid(),
        )
        self.channel_complement = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(encoder_channels * 2, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, encoder_channels, 1),
            nn.Sigmoid(),
        )
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(encoder_channels, encoder_channels, 3, padding=1),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, decoder_feat, encoder_feat):
        decoder_aligned = self.decoder_proj(decoder_feat)
        encoder_proj = self.encoder_proj(encoder_feat)
        feature_diff = torch.abs(encoder_proj - decoder_aligned)
        spatial_weight = self.spatial_complement(torch.cat([encoder_proj, feature_diff], dim=1))
        channel_input = torch.cat(
            [
                F.adaptive_avg_pool2d(encoder_proj, 1),
                F.adaptive_avg_pool2d(feature_diff, 1),
            ],
            dim=1,
        )
        channel_weight = self.channel_complement(channel_input)
        complement_weight = spatial_weight * channel_weight
        enhanced_feat = encoder_feat * (1.0 + complement_weight)
        return self.feature_enhance(enhanced_feat)


# 论文命名别名
DAFM = ComplementarityGuidedFeatureSelection
AFFE = AsymmetricFrequencyFeatureExtraction
FDAW = FrequencyDomainAdaptiveWeightedRegularization


class ESWAModel(nn.Module):
    """
    完整模型（xin6.AdvancedProposedModel 的 ResNet50 路径）。

    空间分支: ResNet50 layer0-3  (64 / 256 / 512 / 1024)
    频域分支: AFFE -> FDAW -> Ffreq，在 layer3 / layer2 与空间特征对齐融合
    Skip:     三处 DAFM (cgfs_e2 / cgfs_e1 / cgfs_e0)
    """

    def __init__(self, pretrained_backbone=True, backbone=None):
        super().__init__()
        if backbone is None:
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            backbone = models.resnet50(weights=weights)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        c0, c1, c2, c3 = 64, 256, 512, 1024

        # ----- 创新点1 AFFE：与 xin6 相同的子模块名，便于加载旧权重 -----
        self.fem = PatchDCTFEM(patch_size=8, high_band_threshold=16, embed_dim=64, num_heads=4)
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # ----- 创新点2 FDAW -----
        self.low_freq_proj = nn.Conv2d(64, 128, kernel_size=1)
        self.difference_regularization = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(64 + 128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.freq_attention = ChannelAttention(128)
        self.fusion_conv = nn.Conv2d(128, c3, kernel_size=1)

        self.fa3 = FeatureAlignment(in_rgb=c3, in_freq=c3, out_channels=c3)
        self.fa2 = FeatureAlignment(in_rgb=c2, in_freq=128, out_channels=c2)

        decoder1_channels, decoder2_channels = 768, 384
        decoder3_channels, decoder4_channels = 192, 96
        self.hor = HORModule(channels=decoder2_channels)

        self.decoder1 = nn.Sequential(
            nn.Conv2d(c3 + c3, decoder1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder1_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder1_channels, decoder1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder1_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(decoder1_channels + c2, decoder2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder2_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder2_channels, decoder2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder2_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(decoder2_channels + c1, decoder3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder3_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder3_channels, decoder3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder3_channels),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(decoder3_channels + c0, decoder4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder4_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder4_channels, decoder4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder4_channels),
            nn.ReLU(inplace=True),
        )

        # ----- 创新点3 DAFM，三处 skip -----
        self.cgfs_e2 = ComplementarityGuidedFeatureSelection(decoder1_channels, c2, reduction=4)
        self.cgfs_e1 = ComplementarityGuidedFeatureSelection(decoder2_channels, c1, reduction=4)
        self.cgfs_e0 = ComplementarityGuidedFeatureSelection(decoder3_channels, c0, reduction=4)

        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder4_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )
        self.relu = nn.ReLU(inplace=True)
        self._initialize_final_layer_bias()

    def _initialize_final_layer_bias(self):
        for m in self.final_conv.modules():
            if isinstance(m, nn.Conv2d) and m.out_channels == 1 and m.bias is not None:
                nn.init.constant_(m.bias, -2.0)

    def _affe(self, x):
        fem_feat = self.fem(x)
        low_map = fem_feat[:, :16, :, :]
        high_map = fem_feat[:, 16:, :, :]
        size = x.shape[2:]
        low_freq_feat = self.low_freq_conv(F.interpolate(low_map, size=size, mode="bilinear", align_corners=True))
        high_freq_feat = self.high_freq_conv(F.interpolate(high_map, size=size, mode="bilinear", align_corners=True))
        return low_freq_feat, high_freq_feat

    def _fdaw(self, low_freq_feat, high_freq_feat):
        low_freq_proj = self.low_freq_proj(low_freq_feat)
        freq_difference = high_freq_feat - low_freq_proj
        regularized_high_freq = self.difference_regularization(freq_difference)
        high_freq_feat = high_freq_feat + 0.3 * regularized_high_freq
        freq_feat = self.freq_fusion(torch.cat([low_freq_feat, high_freq_feat], dim=1))
        freq_feat = self.freq_attention(freq_feat) * freq_feat
        aux = {
            "lfrl": torch.mean(torch.abs(low_freq_feat)),
            "hfrl": torch.mean(torch.abs(regularized_high_freq)),
        }
        return freq_feat, high_freq_feat, aux

    def forward(self, x, return_aux_losses=False):
        e0 = self.layer0(x)
        e1 = self.layer1(e0)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)

        # 创新点1 AFFE
        low_freq_feat, high_freq_feat = self._affe(x)

        # 创新点2 FDAW
        freq_feat, high_freq_feat, aux = self._fdaw(low_freq_feat, high_freq_feat)

        freq_resized = F.interpolate(freq_feat, size=e3.shape[2:], mode="bilinear", align_corners=True)
        freq_fused = self.fusion_conv(freq_resized)
        fused_e3 = self.fa3(e3, freq_fused)

        d1 = self.decoder1(torch.cat([e3, fused_e3], dim=1))
        d1 = F.interpolate(d1, size=e2.shape[2:], mode="bilinear", align_corners=True)

        aligned_e2 = self.fa2(e2, freq_feat)
        e2_enhanced = self.cgfs_e2(decoder_feat=d1, encoder_feat=aligned_e2)
        d2 = self.decoder2(torch.cat([d1, e2_enhanced], dim=1))
        d2 = self.hor(d2)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=True)

        e1_enhanced = self.cgfs_e1(decoder_feat=d2, encoder_feat=e1)
        d3 = self.decoder3(torch.cat([d2, e1_enhanced], dim=1))
        d3 = F.interpolate(d3, size=e0.shape[2:], mode="bilinear", align_corners=True)

        e0_enhanced = self.cgfs_e0(decoder_feat=d3, encoder_feat=e0)
        d4 = self.decoder4(torch.cat([d3, e0_enhanced], dim=1))
        d4 = F.interpolate(d4, size=x.shape[2:], mode="bilinear", align_corners=True)

        output = torch.sigmoid(self.final_conv(d4))
        if return_aux_losses:
            return output, aux
        return output


AdvancedProposedModel = ESWAModel


def build_eswa_model(pretrained_backbone=True, backbone=None) -> nn.Module:
    return ESWAModel(pretrained_backbone=pretrained_backbone, backbone=backbone)


if __name__ == "__main__":
    model = build_eswa_model(pretrained_backbone=False)
    x = torch.randn(1, 3, 352, 352)
    y = model(x)
    print("ESWAModel", tuple(x.shape), "->", tuple(y.shape), "range", float(y.min()), float(y.max()))
    n_params = sum(p.numel() for p in model.parameters())
    print("params", n_params)
