import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']

        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def compute_feat_f_unfold(self, feat_f0, feat_f1, coord0_chunk, coord1_chunk):
        feat_f0_pad = torch.nn.functional.pad(feat_f0, (2, 2, 2, 2), mode='constant')
        feat_f1_pad = torch.nn.functional.pad(feat_f1, (2, 2, 2, 2), mode='constant')
        feat_f0_pad = feat_f0_pad.permute(0, 3, 2, 1)
        feat_f1_pad = feat_f1_pad.permute(0, 3, 2, 1)
        feat_f0_list = []
        feat_f1_list = []
        for delta_y in range(5):
            for delta_x in range(5):
                b0_indexes = coord0_chunk[0].squeeze()
                x0_indexes = coord0_chunk[1].squeeze() + delta_x
                y0_indexes = coord0_chunk[2].squeeze() + delta_y
                feat_f0_list.append(feat_f0_pad[b0_indexes, x0_indexes, y0_indexes])
                b1_indexes = coord1_chunk[0].squeeze()
                x1_indexes = coord1_chunk[1].squeeze() + delta_x
                y1_indexes = coord1_chunk[2].squeeze() + delta_y
                feat_f1_list.append(feat_f1_pad[b1_indexes, x1_indexes, y1_indexes])
        feat_f0_stack = torch.stack(feat_f0_list, -2)
        feat_f1_stack = torch.stack(feat_f1_list, -2)
        if len(feat_f0_stack.size()) < 4:
            feat_f0_stack = feat_f0_stack.unsqueeze(0)
            feat_f1_stack = feat_f1_stack.unsqueeze(0)

        return feat_f0_stack, feat_f1_stack

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, coord0_f, coord1_f, data):
    # def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        data.update({'W': W})
        if data['b_ids'].shape[-1] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1


        # # 1. unfold(crop) all local windows
        # stride = data['hw0_f'][0] // data['hw0_c'][0]
        # feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
        # feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        # feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
        # feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        # # 2. select only the predicted matches # TODO bypass the selection in S2LD Training
        # feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        # feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

        # h0, w0, h1, w1 = data['hw0_f'][0], data['hw0_f'][1], data['hw1_f'][0], data['hw1_f'][1]
        # feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=1, padding=W//2)
        # feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=1, padding=W//2)
        # feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) (h w) -> n h w ww c', ww=W**2, w=w0, h=h0)
        # feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) (h w) -> n h w ww c', ww=W**2, w=w1, h=h1)
        # feat_f0_unfold = feat_f0_unfold.permute(0, 2, 1, 3, 4)[coord0_chunk].squeeze(2)
        # feat_f1_unfold = feat_f1_unfold.permute(0, 2, 1, 3, 4)[coord1_chunk].squeeze(2)

        N = data['image0'].shape[0]
        coord0_long = coord0_f.round().long()
        coord1_long = coord1_f.round().long() # [1, 80, 80, 2]
        batch_index0 = torch.cat([torch.ones([coord0_long.shape[1], coord0_long.shape[2]], device=coord0_f.device)[None, ..., None] * i for i in range(N)], dim=0).long()
        coord0_chunk = rearrange(torch.cat([batch_index0, coord0_long], dim=-1), 'n h w c -> n (h w) c')
        batch_index1 = torch.cat([torch.ones([coord1_long.shape[1], coord1_long.shape[2]], device=coord1_f.device)[None, ..., None] * i for i in range(N)], dim=0).long()
        coord1_chunk = rearrange(torch.cat([batch_index1, coord1_long], dim=-1), 'n h w c -> n (h w) c')

        coord0_chunk = torch.chunk(coord0_chunk, 3, dim=-1)
        coord1_chunk = torch.chunk(coord1_chunk, 3, dim=-1)

        feat_f0_unfold, feat_f1_unfold = self.compute_feat_f_unfold(feat_f0, feat_f1, coord0_chunk, coord1_chunk)

        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

        # option: use coarse-level s2ld feature as context: concat and linear
        if self.cat_c_feat:
            feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                   feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
            feat_cf_win = self.merge_feat(torch.cat([
                torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
            ], -1))
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        return feat_f0_unfold, feat_f1_unfold
