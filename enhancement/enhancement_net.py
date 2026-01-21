import torch
import torch.nn as nn
import torch.nn.functional as F
from .filters import (GammaFilter, ImprovedWhiteBalanceFilter, 
                      DenoiseFilter, DetailEnhancementFilter,
                      ContrastFilter, UsmFilter)
from .util_filters import lrelu
from .config import cfg


class ParameterExtractor(nn.Module):
    """CNN for extracting filter parameters from input image"""

    def __init__(self, output_dim=11, base_channels=16):  # 更新为11个参数
        super(ParameterExtractor, self).__init__()

        # Downsample to 64x64
        self.resize = nn.AdaptiveAvgPool2d(64)

        # Convolutional layers
        self.conv0 = self._make_conv_layer(3, base_channels, stride=2)
        self.conv1 = self._make_conv_layer(base_channels, 2 * base_channels, stride=2)
        self.conv2 = self._make_conv_layer(2 * base_channels, 4 * base_channels, stride=2)
        self.conv3 = self._make_conv_layer(4 * base_channels, 8 * base_channels, stride=2)
        self.conv4 = self._make_conv_layer(8 * base_channels, 8 * base_channels, stride=2)

        # 计算正确的flatten尺寸: 8*base_channels * 2 * 2 = 8*16*4 = 512
        self.fc1 = nn.Linear(8 * base_channels * 2 * 2, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def _make_conv_layer(self, in_ch, out_ch, kernel_size=3, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.resize(x)  # (B, 3, 64, 64)

        x = self.conv0(x)  # (B, 16, 32, 32)
        x = self.conv1(x)  # (B, 32, 16, 16)
        x = self.conv2(x)  # (B, 64, 8, 8)
        x = self.conv3(x)  # (B, 128, 4, 4)
        x = self.conv4(x)  # (B, 128, 2, 2)

        x = x.view(x.size(0), -1)  # (B, 512)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)  # (B, 11)

        return x


class EnhancementModule(nn.Module):
    """
    Complete enhancement module with parameter extraction and 6 filters
    
    滤波器顺序：
    1. Gamma - 伽马校正
    2. WhiteBalance - 白平衡
    3. Denoise - 去噪 (新增)
    4. Contrast - 对比度
    5. DetailEnhancement - 细节增强 (新增)
    6. USM - 锐化
    """

    def __init__(self, use_enhancement=True):
        super(EnhancementModule, self).__init__()
        self.use_enhancement = use_enhancement

        if self.use_enhancement:
            # Parameter extractor
            self.param_extractor = ParameterExtractor(
                output_dim=cfg.num_filter_parameters,
                base_channels=cfg.base_channels
            )

            # Filters (order matters!)
            self.gamma_filter = GammaFilter(cfg)
            self.wb_filter = ImprovedWhiteBalanceFilter(cfg)
            self.denoise_filter = DenoiseFilter(cfg)              # 新增
            self.contrast_filter = ContrastFilter(cfg)
            self.detail_filter = DetailEnhancementFilter(cfg)     # 新增
            self.usm_filter = UsmFilter(cfg)

            self.filters = [
                self.gamma_filter,
                self.wb_filter,
                self.denoise_filter,      # 在对比度前去噪
                self.contrast_filter,
                self.detail_filter,       # 在锐化前增强细节
                self.usm_filter
            ]

    def forward(self, img):
        """
        Args:
            img: (B, 3, H, W) normalized to [0, 1]
        Returns:
            enhanced_img: (B, 3, H, W)
            filter_params: list of parameters for each filter
        """
        if not self.use_enhancement:
            return img, None

        # Extract parameters
        features = self.param_extractor(img)  # (B, 11)

        # Apply filters sequentially
        enhanced = img
        filter_params = []

        for i, filter_module in enumerate(self.filters):
            enhanced, param = filter_module(enhanced, features)
            filter_params.append(param)
            # Clamp after each filter to maintain [0, 1] range
            enhanced = torch.clamp(enhanced, 0, 1)
            
            # 可选：打印每个滤波器的输出范围，用于调试
            print(f"Filter {i} ({filter_module.short_name}): "
                  f"min={enhanced.min().item():.4f}, "
                  f"max={enhanced.max().item():.4f}")

        return enhanced, filter_params


class EnhancementLoss(nn.Module):
    """
    改进的增强损失
    - 减少重建损失权重
    - 添加感知损失鼓励有意义的增强
    """
    def __init__(self, weight=0.01, use_perceptual=True):
        super(EnhancementLoss, self).__init__()
        self.weight = weight
        self.use_perceptual = use_perceptual
    
    def forward(self, enhanced_img, original_img):
        """
        Args:
            enhanced_img: (B, 3, H, W) 增强后图像
            original_img: (B, 3, H, W) 原始图像
        """
        total_loss = 0.0
        
        # 1. 重建损失（大幅降低权重）
        reconstruction_loss = F.mse_loss(enhanced_img, original_img)
        total_loss += self.weight * reconstruction_loss
        
        if self.use_perceptual:
            # 2. 亮度一致性损失（允许亮度变化，但不要过度）
            enhanced_lum = 0.299 * enhanced_img[:, 0] + \
                          0.587 * enhanced_img[:, 1] + \
                          0.114 * enhanced_img[:, 2]
            original_lum = 0.299 * original_img[:, 0] + \
                          0.587 * original_img[:, 1] + \
                          0.114 * original_img[:, 2]
            
            # 只惩罚过度的亮度变化
            lum_diff = torch.abs(enhanced_lum.mean() - original_lum.mean())
            lum_loss = torch.clamp(lum_diff - 0.3, min=0.0)  # 允许30%的亮度变化
            total_loss += 0.1 * lum_loss
            
            # 3. 对比度保持损失（鼓励增加对比度）
            enhanced_std = enhanced_lum.std()
            original_std = original_lum.std()
            # 鼓励对比度增加，但不过度
            contrast_loss = F.relu(original_std - enhanced_std)  # 只惩罚对比度降低
            total_loss += 0.05 * contrast_loss
            
            # 4. 平滑度损失（防止噪声放大）
            # Total Variation loss
            tv_h = torch.abs(enhanced_img[:, :, 1:, :] - enhanced_img[:, :, :-1, :]).mean()
            tv_w = torch.abs(enhanced_img[:, :, :, 1:] - enhanced_img[:, :, :, :-1]).mean()
            tv_loss = tv_h + tv_w
            total_loss += 0.001 * tv_loss
        
        return total_loss
