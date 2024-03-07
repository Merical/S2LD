import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .s2ld_module import LocalFeatureTransformer, FinePreprocess, KeypointDetector, FeatureAdjuster
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching
from .utils.supervision import *


def meshgrid(B, H, W, dtype, device, normalized=False):
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


def image_grid(B, H, W, dtype, device, ones=True, normalized=False):
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    coords = [xs, ys]
    if ones:
        coords.append(torch.ones_like(xs))  # BHW
    grid = torch.stack(coords, dim=1)  # B3HW
    return grid


class S2LDKeypointNet(torch.nn.Module):
    """
    Keypoint detection network.

    Parameters
    ----------
    use_color : bool
        Use color or grayscale images.
    do_upsample: bool
        Upsample desnse descriptor map.
    with_drop : bool
        Use dropout.
    do_cross: bool
        Predict keypoints outside cell borders.
    kwargs : dict
        Extra parameters
    """

    def __init__(self):
        super().__init__()
        backbone_config = {
            "backbone_type": "ResNetFPN",
            "resolution": (8, 2),
            "resnetfpn": {
                'initial_dim': 128,
                'block_dims': [128, 196, 256]
            }
        }

        detector_config = {
            "c_model": 256,
            "f_model": 128,
            "score_dims": [256, 128, 1],
            "keypoint_dims": [256, 128, 2],
            "coarse_dims": [256, 128],
            "fine_dims": [128, 64],
            "resolution": 8,
            "cross_ratio": 1.2
        }

        self.training = True
        self.cross_ratio = 1.2
        self.cell = 8
        self.backbone = build_backbone(backbone_config)
        self.keypoint_detector = KeypointDetector(detector_config)

    def forward(self, x):
        """
        Processes a batch of images.

        Parameters
        ----------
        x : torch.Tensor
            Batch of input images (B, 3, H, W)

        Returns
        -------
        score : torch.Tensor
            Score map (B, 1, H_out, W_out)
        coord: torch.Tensor
            Keypoint coordinates (B, 2, H_out, W_out)
        feat: torch.Tensor
            Keypoint descriptors (B, 256, H_out, W_out)
        """
        B, _, H, W = x.shape
        feat_c0, feat_f0 = self.backbone(x)
        center_shift, score = self.keypoint_detector(feat_c0, feat_f0)  # center_shift already multiplied with the cross_ratio
        return center_shift, score


class KPNet(nn.Module):
    def __init__(self, config, full_config=None, train_kp=True, isolate_kpnet=False):
        super().__init__()
        # Misc
        self.config = config
        self.full_config = full_config
        self.train_kp = train_kp
        self.sthr = 0.4
        self.isolate_kpnet = isolate_kpnet

        # Modules
        self.backbone = build_backbone(config)
        if self.isolate_kpnet:
            self.keypoint_detector = S2LDKeypointNet()
        else:
            self.keypoint_detector = KeypointDetector(config["detector"])

    def compute_warped_kps(self, data, config):
        device = data['image0'].device
        N, _, H0, W0 = data['image0'].shape
        _, _, H1, W1 = data['image1'].shape
        scale = config['resolution'][0]
        scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
        h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
        # h0, w0 = map(lambda x: x // scale, [H0, W0])

        grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)  # [N, hw, 2]
        grid_pt0_i = scale0 * grid_pt0_c
        grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
        grid_pt1_i = scale1 * grid_pt1_c

        kps0 = data["kps0_i_all"]
        kps1 = data["kps1_i_all"]

        grid_pt0_i = grid_pt0_i + kps0
        grid_pt1_i = grid_pt1_i + kps1

        grid_pt0_i = grid_pt0_i.clamp(min=0.)
        grid_pt1_i = grid_pt1_i.clamp(min=0.)

        if 'mask0' in data:
            grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
            grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])
        w_pt0_valid_mask, w_pt0_i_all = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'],
                                                  data['K0'], data['K1'])
        w_pt1_valid_mask, w_pt1_i_all = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'],
                                                  data['K1'], data['K0'])
        data.update({
            "pt0_i_all": grid_pt0_i,
            "w_pt0_i_all": w_pt0_i_all,
            "w_pt0_valid_mask": w_pt0_valid_mask,
            "pt1_i_all": grid_pt1_i,
            "w_pt1_i_all": w_pt1_i_all,
            "w_pt1_valid_mask": w_pt1_valid_mask,
            })

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:],
        })

        if self.isolate_kpnet:
            kps0, score0 = self.keypoint_detector(data['image0'])
        else:
            kps0, score0 = self.keypoint_detector(feat_c0, feat_f0)
        scale0_c2f = (data['hw0_i'][0] / data['hw0_c'][0]) / 2.
        kps0 = kps0 * scale0_c2f
        data.update({"kps0_f_all": kps0, "score0_f_all": score0})
        kps0 = kps0 * data['scale0'].view(data['scale0'].shape[0], 2, 1, 1) if 'scale0' in data else kps0
        kps0 = rearrange(kps0, 'n c h w -> n (h w) c')
        score0 = rearrange(score0, 'n c h w -> n (h w) c').squeeze(-1)
        data.update({"kps0_i_all": kps0, "score0_i_all": score0})

        if self.isolate_kpnet:
            kps1, score1 = self.keypoint_detector(data['image1'])
        else:
            kps1, score1 = self.keypoint_detector(feat_c1, feat_f1)
        scale1_c2f = (data['hw1_i'][0] / data['hw1_c'][0]) / 2.
        kps1 = kps1 * scale1_c2f
        data.update({"kps1_f_all": kps1, "score1_f_all": score1})
        kps1 = kps1 * data['scale1'].view(data['scale1'].shape[0], 2, 1, 1) if 'scale1' in data else kps1
        kps1 = rearrange(kps1, 'n c h w -> n (h w) c')
        score1 = rearrange(score1, 'n c h w -> n (h w) c').squeeze(-1)
        data.update({"kps1_i_all": kps1, "score1_i_all": score1})

        if self.full_config is not None:
            self.compute_warped_kps(data, self.config)



class S2LDKPNet(nn.Module):
    def __init__(self, config, full_config=None, train_kp=True, isolate_kpnet=False, with_filter=False):
        super().__init__()
        # Misc
        self.config = config
        self.full_config = full_config
        self.train_kp = train_kp
        self.sthr = 0.2
        self.isolate_kpnet = isolate_kpnet
        self.with_filter = with_filter

        # Modules
        self.backbone = build_backbone(config)
        if self.isolate_kpnet:
            self.keypoint_detector = S2LDKeypointNet()
            self.feature_adjuster = FeatureAdjuster(config["adjuster"])
        else:
            self.keypoint_detector = KeypointDetector(config["detector"])
            self.feature_adjuster = FeatureAdjuster(config["adjuster"])
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.s2ld_coarse = LocalFeatureTransformer(config['coarse'], level="coarse")
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)
        self.s2ld_fine = LocalFeatureTransformer(config["fine"], level="fine")
        self.fine_matching = FineMatching(config["with_desc"], config["with_kp"])

    def compute_kps_coord_norm(self, data):
        device = data['image0'].device
        N, _, H0, W0 = data['image0'].shape
        _, _, H1, W1 = data['image1'].shape
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale_i2f = data['hw0_f'][0] / data['hw0_i'][0]
        h0, w0, h1, w1 = map(lambda x: int(x // scale), [H0, W0, H1, W1])
        grid_pt0_i = image_grid(N, h0, w0, dtype=data['image0'].dtype, device=device, ones=False,
                                normalized=False) * scale
        grid_pt1_i = image_grid(N, h1, w1, dtype=data['image1'].dtype, device=device, ones=False,
                                normalized=False) * scale

        coord0_i = grid_pt0_i + data["kps0_f_all"]
        coord1_i = grid_pt1_i + data["kps1_f_all"]

        coord0_f = (coord0_i.clone() * scale_i2f).permute(0, 2, 3, 1)
        coord1_f = (coord1_i.clone() * scale_i2f).permute(0, 2, 3, 1)

        # h0, w0, h1, w1 = data['hw0_f'][0], data['hw0_f'][1], data['hw1_f'][0], data['hw1_f'][1]
        coord0_f[:, 0, :, :] = 0.
        coord0_f[:, :, 0, :] = 0.
        coord1_f[:, 0, :, :] = 0.
        coord1_f[:, :, 0, :] = 0.
        coord0_f[:, coord0_f.shape[1]-1, :, :] = 0.
        coord0_f[:, :, coord0_f.shape[2]-1, :] = 0.
        coord1_f[:, coord1_f.shape[1]-1, :, :] = 0.
        coord1_f[:, :, coord1_f.shape[2]-1, :] = 0.

        # h0, w0, h1, w1 = data['hw0_f'][0], data['hw0_f'][1], data['hw1_f'][0], data['hw1_f'][1]
        # coord0_f[..., 0] = coord0_f[..., 0].clamp(min=0., max=float(w0 - 1))
        # coord0_f[..., 1] = coord0_f[..., 1].clamp(min=0., max=float(h0 - 1))
        # coord1_f[..., 0] = coord1_f[..., 0].clamp(min=0., max=float(w1 - 1))
        # coord1_f[..., 1] = coord1_f[..., 1].clamp(min=0., max=float(h1 - 1))

        # coord0_f = coord0_f.clamp(min=0.)
        # coord1_f = coord1_f.clamp(min=0.)

        # coord0_f[..., 0] = coord0_f[..., 0].clamp(min=0.)
        # coord0_f[..., 1] = coord0_f[..., 1].clamp(min=0.)
        # coord1_f[..., 0] = coord1_f[..., 0].clamp(min=0.)
        # coord1_f[..., 1] = coord1_f[..., 1].clamp(min=0.)

        data.update({
            "coords0_i_all": rearrange(coord0_i.clone(), 'n c h w -> n (h w) c'), 
            "coords1_i_all": rearrange(coord1_i.clone(), 'n c h w -> n (h w) c')
            })

        coord0_i = coord0_i.permute(0, 2, 3, 1)
        coord1_i = coord1_i.permute(0, 2, 3, 1)

        coord0_i[..., 0] = coord0_i[..., 0] / (W0 - 1) * 2. - 1.
        coord0_i[..., 1] = coord0_i[..., 1] / (H0 - 1) * 2. - 1.
        coord1_i[..., 0] = coord1_i[..., 0] / (W1 - 1) * 2. - 1.
        coord1_i[..., 1] = coord1_i[..., 1] / (H1 - 1) * 2. - 1.

        return coord0_i, coord1_i, coord0_f, coord1_f

    def compute_warped_kps(self, data, config):
        device = data['image0'].device
        N, _, H0, W0 = data['image0'].shape
        _, _, H1, W1 = data['image1'].shape
        scale = config['resolution'][0]
        scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale
        h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
        # h0, w0 = map(lambda x: x // scale, [H0, W0])

        grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0 * w0, 2).repeat(N, 1, 1)  # [N, hw, 2]
        grid_pt0_i = scale0 * grid_pt0_c
        grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1 * w1, 2).repeat(N, 1, 1)
        grid_pt1_i = scale1 * grid_pt1_c

        kps0 = data["kps0_i_all"]
        kps1 = data["kps1_i_all"]

        grid_pt0_i = grid_pt0_i + kps0
        grid_pt1_i = grid_pt1_i + kps1

        grid_pt0_i = grid_pt0_i.clamp(min=0.)
        grid_pt1_i = grid_pt1_i.clamp(min=0.)

        if 'mask0' in data:
            grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
            grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])
        w_pt0_valid_mask, w_pt0_i_all = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'],
                                                  data['K0'], data['K1'])
        data.update({"w_pt0_i_all": w_pt0_i_all,
                     "pt1_i_all": grid_pt1_i,
                     "w_pt0_valid_mask": w_pt0_valid_mask,
                     })

        if data["b_ids"].shape[0] == 0:
            data.update({
                "mkpts0_f_p_w": data['mkpts0_c_p'],
                "w_valid_mask": None,
                "mkpts1_f_p": data['mkpts1_c_p'],
            })
        else:
            w_pt0_i = w_pt0_i_all[data['b_ids'], data['i_ids']]
            w_valid_mask = w_pt0_valid_mask[data['b_ids'], data['i_ids']]
            mkpts1_f_p = data['mkpts1_c_p'] + data['kps1'] if 'kps1' in data else data['mkpts1_c_p']
            data.update({"mkpts0_f_p_w": w_pt0_i,
                         "w_valid_mask": w_valid_mask,
                         "mkpts1_f_p": mkpts1_f_p,
                         })

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:],
        })

        if self.isolate_kpnet:
            kps0, score0 = self.keypoint_detector(data['image0'])
        else:
            kps0, score0 = self.keypoint_detector(feat_c0, feat_f0)
        feat_c0 = self.feature_adjuster(feat_c0, kps0)
        scale0_c2f = (data['hw0_i'][0] / data['hw0_c'][0]) / 2.
        kps0 = kps0 * scale0_c2f
        data.update({"kps0_f_all": kps0, "score0_f_all": score0})
        kps0 = kps0 * data['scale0'].view(data['scale0'].shape[0], 2, 1, 1) if 'scale0' in data else kps0
        kps0 = rearrange(kps0, 'n c h w -> n (h w) c')
        score0 = rearrange(score0, 'n c h w -> n (h w) c').squeeze(-1)
        data.update({"kps0_i_all": kps0, "score0_i_all": score0})

        if self.isolate_kpnet:
            kps1, score1 = self.keypoint_detector(data['image1'])
        else:
            kps1, score1 = self.keypoint_detector(feat_c1, feat_f1)
        feat_c1 = self.feature_adjuster(feat_c1, kps1)
        scale1_c2f = (data['hw1_i'][0] / data['hw1_c'][0]) / 2.
        kps1 = kps1 * scale1_c2f
        data.update({"kps1_f_all": kps1, "score1_f_all": score1})
        kps1 = kps1 * data['scale1'].view(data['scale1'].shape[0], 2, 1, 1) if 'scale1' in data else kps1
        kps1 = rearrange(kps1, 'n c h w -> n (h w) c')
        score1 = rearrange(score1, 'n c h w -> n (h w) c').squeeze(-1)
        data.update({"kps1_i_all": kps1, "score1_i_all": score1})

        if self.full_config is not None:
            compute_supervision_coarse_kp(data, self.full_config)

        # 2. coarse-level s2ld module
        coord0_norm, coord1_norm, coord0_f, coord1_f = self.compute_kps_coord_norm(data)
        # coord0_norm, coord0_f = compute_kps_coord_norm(data)
        # feat_c0 = self.feature_adjuster(feat_c0, coord0_norm.permute(0, 3, 1, 2))
        # feat_c1 = self.feature_adjuster(feat_c1, coord1_norm.permute(0, 3, 1, 2))
        feat_c0 = self.pos_encoding(feat_c0)
        feat_c1 = self.pos_encoding(feat_c1)
        # feat_c0 = torch.nn.functional.grid_sample(feat_c0, coord0_norm, mode="bilinear",
        #                                           padding_mode="zeros", align_corners=True)
        # feat_c1 = torch.nn.functional.grid_sample(feat_c1, coord1_norm, mode="bilinear",
        #                                           padding_mode="zeros", align_corners=True)
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(feat_c0, 'n c h w -> n (h w) c')
        feat_c1 = rearrange(feat_c1, 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        # feat_c0, feat_c1 = self.s2ld_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_c0, feat_c1 = self.s2ld_coarse(data, feat_c0, feat_c1, mask_c0, mask_c1)
        # data.update({
        #     "ws_c0": weight_self0,
        #     "ws_c1": weight_self1,
        #     "wc_c0": weight_cross0,
        #     "wc_c1": weight_cross1,
        # })

        # 3. match coarse-level
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        if self.config["with_desc"]:
            desc0_indexchunk = torch.cat((data["b_ids"][None, ...], data["i_ids"][None, ...]), 0)
            desc1_indexchunk = torch.cat((data["b_ids"][None, ...], data["j_ids"][None, ...]), 0)
            desc_c0 = feat_c0[desc0_indexchunk.chunk(chunks=2, dim=0)].squeeze()
            desc_c1 = feat_c1[desc1_indexchunk.chunk(chunks=2, dim=0)].squeeze()
            data.update({
                'desc_c0': desc_c0,
                'desc_c1': desc_c1
            })

        # 3.5 select kps and scores according to the coarse_matching
        kps0 = kps0[data['b_ids'], data['i_ids']]
        score0 = score0[data['b_ids'], data['i_ids']]
        data.update({
            'kps0': kps0,
            'score0': score0.squeeze(-1),
        })
        kps1 = kps1[data['b_ids'], data['j_ids']]
        score1 = score1[data['b_ids'], data['j_ids']]
        data.update({
            'kps1': kps1,
            'score1': score1.squeeze(-1),
        })
        if self.full_config is not None:
            self.compute_warped_kps(data, self.config)

        # 4. fine-level refinement
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, coord0_f, coord1_f, data)
        # feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)

        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            # feat_f0_unfold, feat_f1_unfold = self.s2ld_fine(feat_f0_unfold, feat_f1_unfold)
            feat_f0_unfold, feat_f1_unfold = self.s2ld_fine(data, feat_f0_unfold, feat_f1_unfold)


        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

        if self.full_config is None and self.with_filter:
            self.filter_matches(data)

    def filter_matches(self, data):
        kps0 = data['mkpts0_f']
        kps1 = data['mkpts1_f']
        conf = data['mconf']
        score = data['mscore']
        if score.size(0) > 0:
            score_max = score.max()
            score = (score - 0.) / (score_max - 0. + 1e-5)
        mask = score >= self.sthr
        thr_local = self.sthr
        while thr_local >= 0.:
            mask = (score >= thr_local)
            if mask.sum() > 100:
                break
            else:
                thr_local = thr_local - 0.1

        kps0, kps1, conf, score = kps0[mask], kps1[mask], conf[mask], score[mask]
        data.update({
            'mkpts0_f': kps0,
            'mkpts1_f': kps1,
            'mconf': conf,
            'mscore': score,
            'm_bids': data['m_bids'][mask],
            'mmask': data['mmask'][mask]
        })

    def detect(self, data):
        data.update({
            'hw0_i': data['image0'].shape[2:]
        })
        device = data['image0'].device

        (feat_c0, feat_f0) = self.backbone(data['image0'])
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw0_f': feat_f0.shape[2:],
        })
        if self.isolate_kpnet:
            kps0, score0 = self.keypoint_detector(data['image0'])
        else:
            kps0, score0 = self.keypoint_detector(feat_c0, feat_f0)

        scale0_c2f = (data['hw0_i'][0] / data['hw0_c'][0]) / 2.
        kps0 = kps0 * scale0_c2f
        data.update({"kps0_f_all": kps0, "score0_f_all": score0})
        kps0 = kps0 * data['scale0'].view(data['scale0'].shape[0], 2, 1, 1) if 'scale0' in data else kps0

        scale = data["hw0_i"][0] // data["hw0_c"][0]
        scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
        h0, w0 = data["hw0_c"][0], data["hw0_c"][1]

        N = data['image0'].size(0)
        grid_pt0_c = create_meshgrid(h0, w0, False, device)
        grid_pt0_c = grid_pt0_c.permute(0, 3, 1, 2)  # [N, hw, 2]
        grid_pt0_i = scale0 * grid_pt0_c

        kps0_rel = kps0.clone()
        kps0_rel = rearrange(kps0_rel, 'n c h w -> n (h w) c')

        kps0 = grid_pt0_i + kps0
        kps0 = kps0.clamp(min=0.)

        kps0 = rearrange(kps0, 'n c h w -> n (h w) c')
        score0 = rearrange(score0, 'n c h w -> n (h w) c').squeeze(-1)
        return kps0, score0, kps0_rel



if __name__ == '__main__':
    pass