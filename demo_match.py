front_matter = """
------------------------------------------------------------------------
Correspondence estimation demo for [S2LD](https://ieeexplore.ieee.org/document/10159656).

This demo is heavily inspired by [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/) and [LoFTR](https://zju3dv.github.io/loftr/).
We thank the authors for their execellent work.
------------------------------------------------------------------------
"""

import sys
sys.path.append(".")

import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.s2ld_net import S2LDKPNet, default_cfg

torch.set_grad_enabled(False)


def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

def get_cv_kps(pts):
    keypoints = [cv2.KeyPoint(p[0], p[1], 1) for p in pts]
    return keypoints

def get_cv_matches(conf):
    matches = [cv2.DMatch(i, i, (1 - conf[i]) * 10) for i in range(conf.shape[0])]
    return matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='S2LD pair matching demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weight', type=str, help="Path to the checkpoint.")
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    opt = parser.parse_args()
    print(front_matter)
    parser.print_help()

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda'
    isolate_kpnet = False

    matcher = S2LDKPNet(config=default_cfg, isolate_kpnet=isolate_kpnet)
    state_dict = torch.load(opt.weight, map_location='cpu')['state_dict'] if opt.weight.endswith("ckpt") else torch.load(opt.weight, map_location='cpu')
    matcher.load_state_dict(state_dict) 

    matcher = matcher.eval().to(device=device)

    image1 = cv2.imread("./assets/pairs/bear1.png")
    image2 = cv2.imread("./assets/pairs/bear2.png")

    image1 = cv2.resize(image1, tuple(opt.resize))
    image2 = cv2.resize(image2, tuple(opt.resize))

    h_ref, w_ref, h_tgt, w_tgt = image1.shape[0], image1.shape[1], image2.shape[0], image2.shape[1]

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    gray1_tensor = frame2tensor(gray1, device)
    gray2_tensor = frame2tensor(gray2, device)
    last_data = {'image0': gray1_tensor}
    last_data = {**last_data, 'image1': gray2_tensor}

    matcher(last_data)
    margin = 5
    nfeatures = 1024
    thr = 0.5
    conf_thr = 0.8
    total_n_matches = len(last_data['mkpts0_f'])
    mkpts0 = last_data['mkpts0_f'].cpu().numpy()
    mkpts1 = last_data['mkpts1_f'].cpu().numpy()
    mconf = last_data['mconf'].cpu().numpy()
    mscore = last_data['mscore'].cpu().numpy()
    if len(mscore) > 0:
        score_max = mscore.max()
        mscore = (mscore - 0.) / (score_max - 0. + 1e-5)

    mask = (mscore > thr) * (mkpts0[:, 0] <= w_ref - margin) * (mkpts0[:, 1] <= h_ref - margin) * \
           (mkpts0[:, 0] >= margin) * (mkpts0[:, 1] >= margin) * (mkpts1[:, 0] <= w_tgt - margin) * \
           (mkpts1[:, 1] <= h_tgt - margin) * (mkpts1[:, 0] >= margin) * (mkpts1[:, 1] >= margin)
    mask = mask * (mconf > conf_thr)
    mkpts0, mkpts1, mconf, mscore = mkpts0[mask], mkpts1[mask], mconf[mask], mscore[mask]
    mask = np.argsort(mconf)[::-1][0:nfeatures]
    mkpts0, mkpts1, mconf, mscore = mkpts0[mask], mkpts1[mask], mconf[mask], mscore[mask]

    kps1 = get_cv_kps(mkpts0)
    kps2 = get_cv_kps(mkpts1)
    matches = get_cv_matches(mconf)         

    image3 = np.concatenate([image1, image2], axis=1)
    cv2.drawMatches(image1, kps1, image2, kps2, matches, image3, flags=2)
    cv2.imwrite("./demo_matches.png", image3)