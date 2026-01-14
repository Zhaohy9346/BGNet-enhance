from easydict import EasyDict as edict

cfg = edict()

# Filter configuration
cfg.filters = [
    'GammaFilter',
    'ImprovedWhiteBalanceFilter',
    'ContrastFilter',
    'UsmFilter'
]

cfg.num_filter_parameters = 14

# Parameter ranges
cfg.gamma_range = 2.5
cfg.wb_range = 1.1
cfg.contrast_range = (0.0, 1.0)
cfg.usm_range = (0.0, 2.5)

# Parameter begin indices
cfg.gamma_begin_param = 0
cfg.wb_begin_param = 1
cfg.contrast_begin_param = 4
cfg.usm_begin_param = 13

# CNN parameters
cfg.source_img_size = 64
cfg.base_channels = 16
cfg.fc1_size = 64

# Training
cfg.enhancement_loss_weight = 0.1