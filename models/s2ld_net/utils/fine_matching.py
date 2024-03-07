import math
import torch
import torch.nn as nn
import numpy as np

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, with_desc=False, with_kp=False):
        super().__init__()
        self.with_desc = with_desc
        self.with_kp = with_kp

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            data.update({"mkpts1_f_p": data['mkpts1_c_p'],
                         "mscore": data["score0"][data['mmask']]
                         })
            return

        feat_f0_picked = feat_f0[:, WW // 2, :]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)  # size [-1, 5, 5]

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # coords_normalized size [M, 2], heatmap size [batch_size, num_featires, 5, 5]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        # fetch descriptors
        if self.with_desc:
            feat_f1_sample = feat_f1.transpose(2, 1).view(M, C, W, W)
            coords_sample = coords_normalized.view(-1, 1, 1, 2)
            desc_f0 = feat_f0_picked
            desc_f1 = nn.functional.grid_sample(feat_f1_sample, grid=coords_sample, align_corners=True, mode='bilinear').squeeze()
            data.update({'desc0': torch.cat((data['desc_c0'], desc_f0), dim=-1),
                         'desc1': torch.cat((data['desc_c1'], desc_f1), dim=-1),
                         })

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)


    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        mkpts0_f = data['mkpts0_c'] + data['kps0'][data['mmask']] if 'kps0' in data else data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale1' in data else scale
        mkpts1_f = data['mkpts1_c'] + data['kps1'][data['mmask']] + (coords_normed * (W // 2) * scale1)[data['mmask']]

        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
            "mscore": data["score0"][data['mmask']]
        })
        if "mkpts1_c_p" in data:
            mkpts1_f_p = data['mkpts1_c_p'] + data['kps1'] + coords_normed * (W // 2) * scale1
            data.update({"mkpts1_f_p": mkpts1_f_p})