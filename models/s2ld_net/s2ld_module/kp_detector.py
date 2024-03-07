import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, kernel=3, stride=1, dilation=1, bias=False, bn=True, relu=True, sigmoid=False, tanh=False):
    modules = []
    modules.append(nn.Conv2d(inp, oup, kernel, stride, (kernel - 1) * dilation // 2, dilation=dilation, bias=bias))
    if bn:
        modules.append(nn.BatchNorm2d(oup))
    if sigmoid:
        modules.append(nn.Sigmoid())
    elif tanh:
        modules.append(nn.Tanh())
    elif relu:
        modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


class KeypointDetector(nn.Module):
    default_config = {
        "c_model": 256,
        "f_model": 128,
        "score_dims": [256, 1],
        "keypoint_dims": [256, 2],
        "coarse_dims": [256, 256],
        "fine_dims": [256, 256],
        "resolution": 8,
        "cross_ratio": 1.2
    }
    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.c_channels = [self.config["c_model"]] + self.config["coarse_dims"]
        self.f_channels = [self.config["f_model"]] + self.config["fine_dims"]
        self.k_channels = [self.f_channels[-1] + self.c_channels[-1]] + self.config["keypoint_dims"]
        self.s_channels = [self.f_channels[-1] + self.c_channels[-1]] + self.config["score_dims"]

        # self.c_channels = [self.config["c_model"]] + self.config["coarse_dims"]
        # self.f_channels = [self.config["f_model"]] + self.config["fine_dims"]
        # self.k_channels = [self.config["c_model"]] + self.config["keypoint_dims"]
        # self.s_channels = [self.config["c_model"]] + self.config["score_dims"]

        self.resolution = self.config["resolution"]
        self.cross_ratio = self.config["cross_ratio"]

        self.conv_c0 = conv_bn(self.c_channels[0], self.c_channels[1], kernel=3)
        self.conv_c1 = conv_bn(self.c_channels[1], self.c_channels[2], kernel=3)

        self.conv_f0 = conv_bn(self.f_channels[0], self.f_channels[1], kernel=5, stride=2)
        self.conv_f1 = conv_bn(self.f_channels[1], self.f_channels[2], kernel=3, stride=2)

        self.k_regressor = []  # (x, y)
        self.s_regressor = []
        for i, (d_in, d_out) in enumerate(zip(self.k_channels[:-1], self.k_channels[1:])):
            if i == len(self.k_channels[:-1])-1:
                self.k_regressor.append(conv_bn(d_in, d_out, bn=False, tanh=True))
            else:
                self.k_regressor.append(conv_bn(d_in, d_out))
        for i, (d_in, d_out) in enumerate(zip(self.s_channels[:-1], self.s_channels[1:])):
            if i == len(self.s_channels[:-1])-1:
                self.s_regressor.append(conv_bn(d_in, d_out, bn=False, sigmoid=True))
            else:
                self.s_regressor.append(conv_bn(d_in, d_out))
        self.k_regressor = nn.Sequential(*self.k_regressor)
        self.s_regressor = nn.Sequential(*self.s_regressor)

    def forward(self, feat_c, feat_f):
        c0 = self.conv_c0(feat_c)
        c1 = self.conv_c1(c0)
        f0 = self.conv_f0(feat_f)
        f1 = self.conv_f1(f0)

        x = torch.cat([c1, f1], dim=1)
    # def forward(self, x):
        kps = self.k_regressor(x)
        score = self.s_regressor(x)

        kps = kps * self.cross_ratio
        return kps, score


class FeatureAdjuster(nn.Module):
    default_config = {
        "c_model": 256,
        "coarse_dims": [384],
    }
    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.c_channels = [self.config["c_model"]+2] + self.config["coarse_dims"] + [self.config["c_model"]]

        self.c_convs = []  # (x, y)
        for i, (d_in, d_out) in enumerate(zip(self.c_channels[:-1], self.c_channels[1:])):
                self.c_convs.append(conv_bn(d_in, d_out))
        self.c_convs = nn.Sequential(*self.c_convs)

    def forward(self, feat_c, coord_norm):
        x = torch.cat([feat_c, coord_norm], dim=1)
        x = self.c_convs(x)
        return x


class RepeatabilityReason(nn.Module):
    default_config = {
        "d_model": 128,
        "score_dims": [256, 256, 1],
        "resolution": 8,
    }
    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}
        self.input_channel = self.config["d_model"]
        self.s_channels = [self.input_channel] + self.config["score_dims"]

        self.s_regressor = []
        for i, (d_in, d_out) in enumerate(zip(self.s_channels[:-1], self.s_channels[1:])):
            if i == len(self.s_channels[:-1])-1:
                self.s_regressor.append(conv_bn(d_in, d_out, sigmoid=True))
            else:
                self.s_regressor.append(conv_bn(d_in, d_out))
        self.s_regressor = nn.Sequential(*self.s_regressor)

    def forward(self, x):
        score = self.s_regressor(x)
        return score
