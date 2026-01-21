from easydict import EasyDict as edict

cfg = edict()

# Filter configuration - 新增两个滤波器
cfg.filters = [
    'GammaFilter',
    'ImprovedWhiteBalanceFilter',
    'DenoiseFilter',           # 新增
    'ContrastFilter',
    'DetailEnhancementFilter',  # 新增
    'UsmFilter'
]

# 参数总数：1(Gamma) + 3(WB) + 1(Denoise) + 1(Contrast) + 1(Detail) + 1(USM) = 8
# 但考虑到Detail Enhancement可能需要更多参数，我们分配：
# Gamma: 1, WB: 3, Denoise: 2, Contrast: 1, Detail: 3, USM: 1 = 11 参数
cfg.num_filter_parameters = 11

# Parameter ranges
cfg.gamma_range = 2.5
cfg.wb_range = 1.1
cfg.denoise_range = (0.0, 1.0)        # 新增：去噪强度
cfg.contrast_range = (0.0, 1.0)
cfg.detail_range = (0.0, 2.0)         # 新增：细节增强强度
cfg.usm_range = (0.0, 2.5)

# Parameter begin indices (按顺序累加)
cfg.gamma_begin_param = 0              # 1个参数 (0)
cfg.wb_begin_param = 1                 # 3个参数 (1-3)
cfg.denoise_begin_param = 4            # 2个参数 (4-5) - 强度和保护边缘阈值
cfg.contrast_begin_param = 6           # 1个参数 (6)
cfg.detail_begin_param = 7             # 3个参数 (7-9) - 多尺度细节参数
cfg.usm_begin_param = 10               # 1个参数 (10)

# CNN parameters
cfg.source_img_size = 64
cfg.base_channels = 16
cfg.fc1_size = 64

# Training
cfg.enhancement_loss_weight = 0.1
