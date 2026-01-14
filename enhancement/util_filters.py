import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def lrelu(x, leak=0.2):
    """Leaky ReLU activation"""
    return F.leaky_relu(x, negative_slope=leak)

def rgb2lum(image):
    """Convert RGB to luminance"""
    if len(image.shape) == 4:  # (B, C, H, W)
        lum = 0.27 * image[:, 0:1, :, :] + \
              0.67 * image[:, 1:2, :, :] + \
              0.06 * image[:, 2:3, :, :]
    else:  # (B, H, W, C)
        lum = 0.27 * image[:, :, :, 0:1] + \
              0.67 * image[:, :, :, 1:2] + \
              0.06 * image[:, :, :, 2:3]
    return lum

def tanh_range(l, r, initial=None):
    """Tanh range activation"""
    def get_activation(left, right, init):
        def activation(x):
            if init is not None:
                import math
                bias = math.atanh(2 * (init - left) / (right - left) - 1)
            else:
                bias = 0
            return torch.tanh(x + bias) * 0.5 * (right - left) + (right + left) * 0.5
        return activation
    return get_activation(l, r, initial)

def lerp(a, b, l):
    """Linear interpolation"""
    return (1 - l) * a + l * b