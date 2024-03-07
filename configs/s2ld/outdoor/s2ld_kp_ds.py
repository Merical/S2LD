from models.config.default import _CN as cfg

cfg.S2LD.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.S2LD.WITH_KP = True
# cfg.S2LD.WITH_KP = False

cfg.S2LD.LOSS.KEYPOINT_WEIGHT = 0.05  # 0.05 # 20.0
cfg.S2LD.LOSS.CONST_WEIGHT = 0.03  # 20.0
cfg.S2LD.LOSS.SCORE_WEIGHT = 1.0
cfg.S2LD.LOSS.REPEATABILITY_WEIGHT = 1.0  # 2.5
# cfg.S2LD.LOSS.REPEATABILITY_WEIGHT = 0.025 # 2.5

cfg.S2LD.LOSS.DISTRIBUTION_WEIGHT = 0.01
cfg.S2LD.LOSS.SPARSE_WEIGHT = 0.2

cfg.TRAINER.CANONICAL_LR = 8e-3
# cfg.TRAINER.CANONICAL_LR = 8e-9
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
# cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]
cfg.TRAINER.MSLR_MILESTONES = [6, 10, 14, 18, 22, 26, 29, 32, 35, 38]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.TRAINER.ENABLE_PLOTTING = False
cfg.S2LD.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
