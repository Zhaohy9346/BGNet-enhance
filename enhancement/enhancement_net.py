import torch
import torch.nn as nn
import torch.nn.functional as F
from .filters import GammaFilter, ImprovedWhiteBalanceFilter, ContrastFilter, UsmFilter
from .util_filters import lrelu
from .config import cfg


class ParameterExtractor(nn.Module):
    """CNN for extracting filter parameters from input image"""

    def __init__(self, output_dim=14, base_channels=16):
        super(ParameterExtractor, self).__init__()

        # Downsample to 64x64
        self.resize = nn.AdaptiveAvgPool2d(64)

        # Convolutional layers
        self.conv0 = self._make_conv_layer(3, base_channels, stride=2)
        self.conv1 = self._make_conv_layer(base_channels, 2 * base_channels, stride=2)
        self.conv2 = self._make_conv_layer(2 * base_channels, 2 * base_channels, stride=2)
        self.conv3 = self._make_conv_layer(2 * base_channels, 2 * base_channels, stride=2)
        self.conv4 = self._make_conv_layer(2 * base_channels, 2 * base_channels, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(2048, 64)
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
        x = self.conv2(x)  # (B, 32, 8, 8)
        x = self.conv3(x)  # (B, 32, 4, 4)
        x = self.conv4(x)  # (B, 32, 2, 2)

        x = x.view(x.size(0), -1)  # (B, 2048)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)  # (B, 14)

        return x


class EnhancementModule(nn.Module):
    """Complete enhancement module with parameter extraction and filters"""

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
            self.contrast_filter = ContrastFilter(cfg)
            self.usm_filter = UsmFilter(cfg)

            self.filters = [
                self.gamma_filter,
                self.wb_filter,
                self.contrast_filter,
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
        features = self.param_extractor(img)  # (B, 14)

        # Apply filters sequentially
        enhanced = img
        filter_params = []

        for filter_module in self.filters:
            enhanced, param = filter_module(enhanced, features)
            filter_params.append(param)
            enhanced = torch.clamp(enhanced, 0, 1)

        return enhanced, filter_params


class EnhancementLoss(nn.Module):
    """Loss for enhancement module"""

    def __init__(self, weight=0.1):
        super(EnhancementLoss, self).__init__()
        self.weight = weight

    def forward(self, enhanced_img, original_img):
        """
        Reconstruction loss to prevent over-enhancement
        """
        # L2 loss
        reconstruction_loss = F.mse_loss(enhanced_img, original_img)

        return self.weight * reconstruction_loss