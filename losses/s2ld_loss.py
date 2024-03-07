from loguru import logger

import torch
import torch.nn as nn
from models.s2ld_net.utils.geometry import warp_kpts
from einops import repeat, rearrange


class S2LDLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['s2ld']['loss']
        self.match_type = self.config['s2ld']['match_coarse']['match_type']
        self.sparse_spvs = self.config['s2ld']['match_coarse']['sparse_spvs']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def get_uni_xy(self, position):
        idx = torch.argsort(position)
        idx = idx.float()
        p = position.shape[0]
        uni_l2 = torch.mean(torch.pow(position - (idx / p), 2))
        return uni_l2

    def compute_keypoint_distribution_loss(self, pt0, pt1, bids, bs):
        loss = torch.scalar_tensor(0., device=pt0.device)
        for b in range(bs):
            pt0_local = pt0[bids == b]
            pt1_local = pt1[bids == b]
            loss += self.get_uni_xy(pt0_local[:, 0])
            loss += self.get_uni_xy(pt0_local[:, 1])
            loss += self.get_uni_xy(pt1_local[:, 0])
            loss += self.get_uni_xy(pt1_local[:, 1])
        loss = loss / (bs*4)
        return loss

    def compute_keypoint_repeatability_loss(self, pt0, pt1, score0, score1, wh0, wh1, mask=None):

        if not pt0.size(0):
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                logger.warning("keypoint repeatability:: avoid ddp deadlock")
                return torch.scalar_tensor(0., device=pt0.device)
        pt0_norm = pt0
        pt1_norm = pt1
        score0 = score0.unsqueeze(-1)
        score1 = score1.unsqueeze(-1)

        if mask is not None and mask.shape[0] != 0:
            pt0_norm = pt0_norm[mask]
            pt1_norm = pt1_norm[mask]
            score0 = score0[mask]
            score1 = score1[mask]

        mask = pt0_norm.abs() <= 0.5
        mask = mask[:, 0] * mask[:, 1]
        if mask.sum() > 10:
            pt0_norm = pt0_norm[mask]
            pt1_norm = pt1_norm[mask]
            score0 = score0[mask]
            score1 = score1[mask]

        dist = ((pt0_norm - pt1_norm) ** 2).sum(dim=-1).sqrt()
        dist_mean = dist.mean()
        score_mean = (score0 + score1)/2
        loss = (dist - dist_mean) * score_mean

        loss = loss.mean()

        isnan = torch.isnan(loss).detach().cpu().numpy()
        if isnan:
            loss = torch.scalar_tensor(0., device=loss.device)

        return loss

    def compute_keypoint_l2_loss(self, pt0, pt1, wh0, wh1, mask=None):
        pt0_norm = pt0
        pt1_norm = pt1

        if mask is not None and mask.shape[0] != 0:
            pt0_norm = pt0_norm[mask]
            pt1_norm = pt1_norm[mask]

        # Only consider pts in the range
        mask = pt0_norm.abs() <= 0.5
        mask = mask[:, 0] * mask[:, 1]
        if mask.sum() > 10:
            pt0_norm = pt0_norm[mask]
            pt1_norm = pt1_norm[mask]

        dist = ((pt0_norm - pt1_norm) ** 2).sum(dim=-1).sqrt()
        loss = dist.mean()

        isnan = torch.isnan(loss).detach().cpu().numpy()
        if isnan:
            loss = torch.scalar_tensor(0., device=loss.device)

        return loss

    def compute_keypoint_loss(self, data):
        # localization loss
        src_pts_w = data["w_pt0_i_all"]
        tgt_pts = data["pt1_i_all"]
        valid_mask = data["w_pt0_valid_mask"]

        if "mask0" in data:
            mask0 = repeat(data["mask0"], 'n h w -> n (h w) c', c=2)
            valid_mask = valid_mask & mask0[..., 0] & mask0[..., 1]

        thr = (8 * data['scale0'][:, [0]]) if 'scale0' in data else 8
        loc_mat_abs = torch.abs(src_pts_w.unsqueeze(2) - tgt_pts.unsqueeze(1))
        loc_mat_l2 = torch.norm(loc_mat_abs, p=2, dim=3)
        loc_mat_l2_min, loc_mat_l2_min_index = loc_mat_l2.min(dim=2)
        dist_valid_mask = (loc_mat_l2_min <= thr) & valid_mask

        loc_loss = loc_mat_l2_min[dist_valid_mask]
        loc_loss_mean = loc_loss.mean()

        # repeatability loss
        src_score = data["score0_i_all"]
        tgt_score = data["score1_i_all"]
        tgt_score_ass = tgt_score.gather(1, loc_mat_l2_min_index)
        rep_loss = (tgt_score_ass[dist_valid_mask] + src_score[dist_valid_mask])
        rep_loss = rep_loss * (loc_loss - loc_loss_mean)
        rep_loss_mean = rep_loss.mean()

        N, _, H1, W1 = data['image1'].shape
        if 'scale1' in data:
            W1 = W1 * data['scale1'][..., 0]
            H1 = H1 * data['scale1'][..., 1]

        # score consistency loss
        src_pts_w_norm = src_pts_w.clone()
        src_pts_w_norm[..., 0] = src_pts_w_norm[..., 0] / ((W1*8-1) / 2.).view(W1.shape[0], 1) - 1
        src_pts_w_norm[..., 1] = src_pts_w_norm[..., 1] / ((H1*8-1) / 2.).view(H1.shape[0], 1) - 1
        tgt_score_mat = rearrange(tgt_score, "n (h w) -> n h w", h=data['hw1_c'][0], w=data['hw1_c'][1]).unsqueeze(1)
        src_pts_w_norm = rearrange(src_pts_w_norm,  "n (h w) c-> n h w c", h=data['hw0_c'][0], w=data['hw0_c'][1])
        tgt_score_resampled = torch.nn.functional.grid_sample(tgt_score_mat, src_pts_w_norm, mode='bilinear', align_corners=True)
        tgt_score_resampled = rearrange(tgt_score_resampled.squeeze(1), "n h w -> n (h w)")
        score_loss = torch.nn.functional.mse_loss(tgt_score_resampled[valid_mask], src_score[valid_mask]).mean() * 2

        return loc_loss_mean, rep_loss_mean, score_loss

    def compute_keypoint_score_loss(self, score0, score1, mask=None):
        if mask is not None and mask.shape[0] != 0:
            score0 = score0[mask]
            score1 = score1[mask]
        loss = (score0 - score1) ** 2
        loss = loss.mean()
        return loss

    def inverse_huber_loss(self, diff):
        if diff.numel() == 0:
            return torch.scalar_tensor(0., device=diff.device)
        else:
            absdiff = torch.abs(diff)
            C = 0.2 * torch.max(absdiff).item()
            return torch.mean(torch.where(absdiff < C, absdiff, (absdiff * absdiff + C * C) / (2 * C)))

    def compute_depth_sparse_loss(self, keypoints, spv, depth_gt_map, bselect):
        keypoints_index = keypoints.long()
        bselect = bselect.unsqueeze(1)
        keypoints_indexchunk = torch.cat((bselect, keypoints_index[:,[1]], keypoints_index[:,[0]]), 1)
        keypoints_indexchunk = keypoints_indexchunk.view(-1,3).t()
        dgv = depth_gt_map[keypoints_indexchunk.chunk(chunks=3, dim=0)].squeeze()

        if not spv.dim() == dgv.dim():
            return torch.scalar_tensor(0., device=keypoints.device)
        valid_mask = (dgv > 0).detach()
        diff = spv - dgv
        diff = diff[valid_mask]
        loss = self.inverse_huber_loss(diff)
        return loss

    def compute_depth_loss(self, depth, depth_gt):
        # MSE L2 loss
        assert depth.dim() == depth_gt.dim(), "S2LD_LOSS: inconsistent dimensions in compute_depth_loss"

        valid_mask = (depth_gt > 0).detach()
        diff = depth_gt - depth
        diff = diff[valid_mask]
        loss = (diff ** 2).mean()
        loss = torch.scalar_tensor(0., device=loss.device) if torch.isnan(loss) else loss
        return loss

    def compute_depth_sparse_loss_full(self, depth, depth_gt):
        # MSE L2 loss
        assert depth.dim() == depth_gt.dim(), "S2LD_LOSS: inconsistent dimensions in compute_depth_sparse_loss"
        valid_mask = ((depth_gt > 0) & (depth > 0)).detach()
        diff = depth_gt - depth
        diff = diff[valid_mask]
        loss = (diff ** 2).mean()
        loss = torch.scalar_tensor(0., device=loss.device) if torch.isnan(loss) else loss
        return loss

    def compute_depth_regularization(self, x):
        # depth regularization
        assert x.dim(
        ) == 4, "expected 4-dimensional data, but instead got {}".format(
            x.dim())
        horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :-2] - x[:, :, 1:-1, 2:]
        vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:-1] - x[:, :, 2:, 1:-1]
        der_2nd = horizontal.abs() + vertical.abs()
        regu = der_2nd.mean()
        regu = torch.scalar_tensor(0., device=regu.device)  if torch.isnan(regu) else regu
        return regu

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                
                loss = c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))
        
    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        loss = torch.scalar_tensor(0., device=data['image0'].device)

        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        loss_c = self.compute_coarse_loss(
            data['conf_matrix_with_bin'] if self.sparse_spvs and self.match_type == 'sinkhorn' \
                else data['conf_matrix'],
            data['conf_matrix_gt'],
            weight=c_weight)
        loss = loss_c * self.loss_config['coarse_weight']
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level loss
        loss_f = self.compute_fine_loss(data['expec_f'], data['expec_f_gt'])
        if loss_f is not None:
            loss += loss_f * self.loss_config['fine_weight']
            loss_scalars.update({"loss_f":  loss_f.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_f': torch.tensor(1.)})  # 1 is the upper bound

        # 3. keypoint losses
        loss_k, loss_kr, loss_ks = self.compute_keypoint_loss(data)
        loss_k = loss_k * self.loss_config['keypoint_weight']
        loss_kr = loss_kr * self.loss_config['repeatability_weight']
        loss_ks = loss_ks * self.loss_config['score_weight']
        loss += loss_k
        loss += loss_kr
        loss += loss_ks
        loss_scalars.update({'loss_k': loss_k.clone().detach().cpu()})
        loss_scalars.update({'loss_kr': loss_kr.clone().detach().cpu()})
        loss_scalars.update({'loss_ks': loss_ks.clone().detach().cpu()})
        loss_scalars.update({'score_mean': data['score0'].mean().clone().detach().cpu()})

        # 4. sparse-depth loss
        loss_s = self.compute_depth_sparse_loss(data["mkpts0_f"], data["depth0_sparse_value"], data['depth0'], data['m_bids'])
        if loss_s is not None:
            loss += loss_s * self.loss_config['sparse_weight']
            loss_scalars.update({"loss_s":  loss_s.clone().detach().cpu()})
        else:
            assert self.training is False
            loss_scalars.update({'loss_s': torch.tensor(1.)})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
