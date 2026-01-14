import torch
import torch.nn as nn
import torch.nn.functional as F
from .util_filters import *
import math


class Filter(nn.Module):
    """Base filter class"""

    def __init__(self, cfg):
        super(Filter, self).__init__()
        self.cfg = cfg
        self.num_filter_parameters = None
        self.short_name = None
        self.begin_filter_parameter = None

    def get_num_filter_parameters(self):
        return self.num_filter_parameters

    def get_begin_filter_parameter(self):
        return self.begin_filter_parameter

    def filter_param_regressor(self, features):
        """Extract filter-specific parameters from features"""
        raise NotImplementedError

    def process(self, img, param):
        """Apply filter to image"""
        raise NotImplementedError

    def forward(self, img, features):
        """Apply filter with learned parameters"""
        filter_params = self.filter_param_regressor(features)
        return self.process(img, filter_params), filter_params


class GammaFilter(Filter):
    """Gamma correction filter"""

    def __init__(self, cfg):
        super(GammaFilter, self).__init__(cfg)
        self.short_name = 'G'
        self.begin_filter_parameter = cfg.gamma_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        log_gamma_range = math.log(self.cfg.gamma_range)
        param = features[:, self.begin_filter_parameter:self.begin_filter_parameter + 1]
        return torch.exp(tanh_range(-log_gamma_range, log_gamma_range)(param))

    def process(self, img, param):
        # img: (B, C, H, W)
        param = param.view(-1, 1, 1, 1).expand(-1, 3, -1, -1)
        return torch.pow(torch.clamp(img, 0.001, 1.0), param)


class ImprovedWhiteBalanceFilter(Filter):
    """White balance filter"""

    def __init__(self, cfg):
        super(ImprovedWhiteBalanceFilter, self).__init__(cfg)
        self.short_name = 'WB'
        self.channels = 3
        self.begin_filter_parameter = cfg.wb_begin_param
        self.num_filter_parameters = self.channels

    def filter_param_regressor(self, features):
        log_wb_range = 0.5
        param = features[:, self.begin_filter_parameter:self.begin_filter_parameter + 3]

        # Mask: only adjust G and B channels
        mask = torch.tensor([0, 1, 1], dtype=torch.float32, device=param.device)
        param = param * mask

        color_scaling = torch.exp(tanh_range(-log_wb_range, log_wb_range)(param))

        # Normalize by luminance
        lum = 1e-5 + 0.27 * color_scaling[:, 0] + \
              0.67 * color_scaling[:, 1] + \
              0.06 * color_scaling[:, 2]
        color_scaling = color_scaling / lum.unsqueeze(1)

        return color_scaling

    def process(self, img, param):
        # param: (B, 3)
        param = param.view(-1, 3, 1, 1)
        return img * param


class ContrastFilter(Filter):
    """Contrast adjustment filter"""

    def __init__(self, cfg):
        super(ContrastFilter, self).__init__(cfg)
        self.short_name = 'Ct'
        self.begin_filter_parameter = cfg.contrast_begin_param
        self.num_filter_parameters = 1

    def filter_param_regressor(self, features):
        param = features[:, self.begin_filter_parameter:self.begin_filter_parameter + 1]
        return torch.tanh(param)

    def process(self, img, param):
        # Convert to luminance
        luminance = torch.clamp(rgb2lum(img), 0.0, 1.0)

        # Apply contrast curve
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum

        # Lerp between original and contrast
        param = param.view(-1, 1, 1, 1)
        return lerp(img, contrast_image, param)


class UsmFilter(Filter):
    """Unsharp masking filter for sharpening"""

    def __init__(self, cfg):
        super(UsmFilter, self).__init__(cfg)
        self.short_name = 'USM'
        self.begin_filter_parameter = cfg.usm_begin_param
        self.num_filter_parameters = 1

        # Pre-create Gaussian kernel
        self.register_buffer('kernel', self._make_gaussian_kernel())

    def _make_gaussian_kernel(self, sigma=5.0, radius=12):
        """Create 2D Gaussian kernel"""
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        kernel = k.view(1, -1) * k.view(-1, 1)
        return kernel.view(1, 1, 2 * radius + 1, 2 * radius + 1)

    def filter_param_regressor(self, features):
        param = features[:, self.begin_filter_parameter:self.begin_filter_parameter + 1]
        return tanh_range(*self.cfg.usm_range)(param)

    def process(self, img, param):
        # Apply Gaussian blur to each channel
        kernel = self.kernel.repeat(3, 1, 1, 1)  # (3, 1, 25, 25)

        # Pad image
        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')

        # Convolve each channel separately
        blurred = F.conv2d(padded, kernel, groups=3, padding=0)

        # Unsharp mask: img + strength * (img - blurred)
        param = param.view(-1, 1, 1, 1)
        return img + param * (img - blurred)