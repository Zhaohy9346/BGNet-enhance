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
        param = param.view(-1, 3, 1, 1)
        return img * param


class DenoiseFilter(Filter):
    """
    自适应去噪滤波器
    使用双边滤波思想：保留边缘，平滑噪声
    """
    def __init__(self, cfg):
        super(DenoiseFilter, self).__init__(cfg)
        self.short_name = 'DN'
        self.begin_filter_parameter = cfg.denoise_begin_param
        self.num_filter_parameters = 2  # 去噪强度 + 边缘保护阈值
        
        # 创建高斯核用于空间平滑
        self.register_buffer('spatial_kernel', self._make_gaussian_kernel(sigma=1.5, radius=3))
    
    def _make_gaussian_kernel(self, sigma=1.5, radius=3):
        """创建高斯核"""
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        kernel = k.view(1, -1) * k.view(-1, 1)
        return kernel.view(1, 1, 2 * radius + 1, 2 * radius + 1)
    
    def filter_param_regressor(self, features):
        param = features[:, self.begin_filter_parameter:self.begin_filter_parameter + 2]
        # 去噪强度: [0, 1]
        denoise_strength = torch.sigmoid(param[:, 0:1])
        # 边缘保护阈值: [0.1, 0.5]
        edge_threshold = tanh_range(0.1, 0.5)(param[:, 1:2])
        return torch.cat([denoise_strength, edge_threshold], dim=1)
    
    def process(self, img, param):
        """
        自适应双边滤波去噪
        Args:
            img: (B, 3, H, W)
            param: (B, 2) - [强度, 边缘阈值]
        """
        strength = param[:, 0:1].view(-1, 1, 1, 1)
        edge_thresh = param[:, 1:2].view(-1, 1, 1, 1)
        
        # 计算梯度用于边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2)
        
        # 对每个通道计算梯度
        grad_x = F.conv2d(img, sobel_x.repeat(3, 1, 1, 1), padding=1, groups=3)
        grad_y = F.conv2d(img, sobel_y.repeat(3, 1, 1, 1), padding=1, groups=3)
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 边缘mask：边缘区域不去噪
        edge_mask = (edge_map < edge_thresh).float()
        
        # 应用高斯平滑
        kernel = self.spatial_kernel.repeat(3, 1, 1, 1)
        pad_w = (self.spatial_kernel.size(-1) - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')
        smoothed = F.conv2d(padded, kernel, groups=3, padding=0)
        
        # 根据强度和边缘mask混合
        denoised = img * (1 - strength * edge_mask) + smoothed * strength * edge_mask
        
        return torch.clamp(denoised, 0, 1)


class DetailEnhancementFilter(Filter):
    """
    多尺度细节增强滤波器
    提取并增强中频细节（纹理），不同于USM的高频锐化
    """
    def __init__(self, cfg):
        super(DetailEnhancementFilter, self).__init__(cfg)
        self.short_name = 'DE'
        self.begin_filter_parameter = cfg.detail_begin_param
        self.num_filter_parameters = 3  # 3个尺度的细节增强强度
        
        # 创建多尺度高斯核
        self.register_buffer('kernel_fine', self._make_gaussian_kernel(sigma=1.0, radius=3))
        self.register_buffer('kernel_medium', self._make_gaussian_kernel(sigma=2.5, radius=5))
        self.register_buffer('kernel_coarse', self._make_gaussian_kernel(sigma=5.0, radius=7))
    
    def _make_gaussian_kernel(self, sigma, radius):
        """创建高斯核"""
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        k = k / k.sum()
        kernel = k.view(1, -1) * k.view(-1, 1)
        return kernel.view(1, 1, 2 * radius + 1, 2 * radius + 1)
    
    def filter_param_regressor(self, features):
        param = features[:, self.begin_filter_parameter:self.begin_filter_parameter + 3]
        # 3个尺度的增强强度: [0, 2]
        return tanh_range(*self.cfg.detail_range)(param)
    
    def process(self, img, param):
        """
        多尺度细节增强
        Args:
            img: (B, 3, H, W)
            param: (B, 3) - 3个尺度的强度
        """
        fine_strength = param[:, 0:1].view(-1, 1, 1, 1)
        medium_strength = param[:, 1:2].view(-1, 1, 1, 1)
        coarse_strength = param[:, 2:3].view(-1, 1, 1, 1)
        
        # 提取三个尺度的细节
        def extract_detail(img, kernel):
            kernel_3ch = kernel.repeat(3, 1, 1, 1)
            pad_w = (kernel.size(-1) - 1) // 2
            padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')
            blurred = F.conv2d(padded, kernel_3ch, groups=3, padding=0)
            detail = img - blurred
            return detail
        
        detail_fine = extract_detail(img, self.kernel_fine)
        detail_medium = extract_detail(img, self.kernel_medium)
        detail_coarse = extract_detail(img, self.kernel_coarse)
        
        # 加权组合细节
        enhanced = img + \
                   fine_strength * detail_fine + \
                   medium_strength * detail_medium + \
                   coarse_strength * detail_coarse
        
        return torch.clamp(enhanced, 0, 1)


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
        kernel = self.kernel.repeat(3, 1, 1, 1)
        
        # Pad image
        pad_w = (25 - 1) // 2
        padded = F.pad(img, (pad_w, pad_w, pad_w, pad_w), mode='reflect')
        
        # Convolve each channel separately
        blurred = F.conv2d(padded, kernel, groups=3, padding=0)
        
        # Unsharp mask: img + strength * (img - blurred)
        param = param.view(-1, 1, 1, 1)
        return img + param * (img - blurred)
