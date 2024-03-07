from yacs.config import CfgNode as CN


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()
_CN.BACKBONE_TYPE = 'ResNetFPN'
_CN.RESOLUTION = (8, 2)  # options: [(8, 2), (16, 4)]
_CN.FINE_WINDOW_SIZE = 5  # window_size in fine_level, must be odd
_CN.FINE_CONCAT_COARSE_FEAT = True
_CN.WITH_DESC = False
_CN.WITH_KP = True

# 1. S2LD-backbone (local feature CNN) config
_CN.RESNETFPN = CN()
_CN.RESNETFPN.INITIAL_DIM = 128
_CN.RESNETFPN.BLOCK_DIMS = [128, 196, 256]  # s1, s2, s3

# 2. S2LD-coarse module config
_CN.COARSE = CN()
_CN.COARSE.D_MODEL = 256
_CN.COARSE.D_FFN = 256
_CN.COARSE.NHEAD = 8
_CN.COARSE.LAYER_NAMES = ['self', 'cross'] * 4
_CN.COARSE.ATTENTION = 'linear'  # options: ['linear', 'full']

# 3. Coarse-Matching config
_CN.MATCH_COARSE = CN()
_CN.MATCH_COARSE.THR = 0.2
# _CN.MATCH_COARSE.THR = 0.1
# _CN.MATCH_COARSE.THR = 0.7
_CN.MATCH_COARSE.BORDER_RM = 2
_CN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'  # options: ['dual_softmax, 'sinkhorn']
_CN.MATCH_COARSE.DSMAX_TEMPERATURE = 0.1
_CN.MATCH_COARSE.SKH_ITERS = 3
_CN.MATCH_COARSE.SKH_INIT_BIN_SCORE = 1.0
_CN.MATCH_COARSE.SKH_PREFILTER = True
_CN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.4  # training tricks: save GPU memory
_CN.MATCH_COARSE.TRAIN_PAD_NUM_GT_MIN = 200  # training tricks: avoid DDP deadlock

# 4. S2LD-fine module config
_CN.FINE = CN()
_CN.FINE.D_MODEL = 128
_CN.FINE.D_FFN = 128
_CN.FINE.NHEAD = 8
_CN.FINE.LAYER_NAMES = ['self', 'cross'] * 1
_CN.FINE.ATTENTION = 'linear'

# 5. Optical Flow Encoding config
_CN.FLOW = CN()
_CN.FLOW.HEIGHT = 480
_CN.FLOW.WIDTH = 640

# 6. Keypoint Detector Module Config
_CN.DETECTOR = CN()
_CN.DETECTOR.C_MODEL = 256
_CN.DETECTOR.F_MODEL = 128
_CN.DETECTOR.COARSE_DIMS = [256, 128]
_CN.DETECTOR.FINE_DIMS = [128, 64]
_CN.DETECTOR.SCORE_DIMS = [256, 128, 1]
_CN.DETECTOR.KEYPOINT_DIMS = [256, 128, 2]
_CN.DETECTOR.RESOLUTION = 8

_CN.ADJUSTER = CN()
_CN.ADJUSTER.C_MODEL = 256
_CN.ADJUSTER.COARSE_DIMS = [384]

default_cfg = lower_config(_CN)
