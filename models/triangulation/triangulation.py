import torch
import torch.nn as nn
import numpy as np
import time

def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")

def triangulate_point_from_multiple_views_linear_torch_batch(proj_matricies, points, confidences=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """

    assert len(proj_matricies) == len(points), print("proj_matricies shape is {}, points shape is {}".format(proj_matricies.shape,
                                                                                                             points.shape))
    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(points.shape[1], n_views, dtype=torch.float32, device=points.device)

    ##multiple points
    points_t = points.transpose(0,1)
    proj_mat = proj_matricies[:, 2:3].expand(n_views, 2, 4).unsqueeze(0)
    points_tview = points_t.view(points_t.size(0), n_views, 2, 1).expand(points_t.size(0), n_views, 2, 4)
    A_all = proj_mat * points_tview - proj_matricies[:, :2].unsqueeze(0)
    A_all *= confidences.view(confidences.size(0), n_views, 1, 1)
    A_all = A_all.contiguous().view(A_all.size(0), A_all.size(1)*A_all.size(2), 4)
    _, _, V = torch.svd(A_all)
    points_3d = homogeneous_to_euclidean(-V[:,:, 3])

    return points_3d

def triangulate_batch_of_points(proj_matricies_batch, points_batch, confidences_batch=None):
    """Triangulates for a batch of points"""
    batch_size, n_views = proj_matricies_batch.shape[:2]

    points_3d_batch = []
    for batch_i in range(batch_size):
        n_points = points_batch[batch_i].shape[1]
        points = points_batch[batch_i]
        confidences = confidences_batch[batch_i] if confidences_batch is not None else None
        points_3d = triangulate_point_from_multiple_views_linear_torch_batch(proj_matricies_batch[batch_i], points, confidences=confidences)
        points_3d_batch.append(points_3d)

    return points_3d_batch


def integrate_tensor_2d(heatmaps, softmax=True): #,temperature = 1.0):
    """Applies softmax to heatmaps and integrates them to get their's "center of masses"
    Args:
        heatmaps torch tensor of shape (batch_size, n_heatmaps, h, w): input heatmaps
    Returns:
        coordinates torch tensor of shape (batch_size, n_heatmaps, 2): coordinates of center of masses of all heatmaps
    """
    batch_size, n_heatmaps, h, w = heatmaps.shape

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, -1))
    
    if softmax:
        heatmaps = torch.nn.functional.softmax(heatmaps, dim=2)
    else:
        heatmaps = torch.nn.functional.relu(heatmaps)

    heatmaps = heatmaps.reshape((batch_size, n_heatmaps, h, w))

    mass_x = heatmaps.sum(dim=2)
    mass_y = heatmaps.sum(dim=3)

    mass_times_coord_x = mass_x * torch.arange(w).type(torch.float).to(mass_x.device)
    mass_times_coord_y = mass_y * torch.arange(h).type(torch.float).to(mass_y.device)

    x = mass_times_coord_x.sum(dim=2, keepdim=True)
    y = mass_times_coord_y.sum(dim=2, keepdim=True)


    if not softmax:
        x = x / mass_x.sum(dim=2, keepdim=True)
        y = y / mass_y.sum(dim=2, keepdim=True)

    coordinates = torch.cat((x, y), dim=2)
    coordinates = coordinates.reshape((batch_size, n_heatmaps, 2))

    return coordinates


def unproject_ij(keypoints_2d, z, camera_matrix):
    """Unprojects points into 3D using intrinsics""" 

    z = z.squeeze(2).squeeze(1)
    x =  ((keypoints_2d[:,:, 0] - camera_matrix[:,[0], [2]]) / camera_matrix[:,[0], [0]])*z
    y = ((keypoints_2d[:,:, 1] - camera_matrix[:,[1], [2]]) / camera_matrix[:,[1], [1]])*z
    xyz = torch.cat((x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)), dim=1)
    return xyz


def reproject_points(pose, pts, K0, K1, Z):
    """Projects 3d points onto 2D image plane"""

    kp_arr = torch.ones((pts.shape[0],pts.shape[1],3)).to(pts.device)
    kp_arr[:,:,:2] = pts

    K0 = K0.unsqueeze(1)
    K1 = K1.unsqueeze(0)
    # K = torch.cat((K0.unsqueeze(1), K1.unsqueeze(1)), dim=1)  # TODO: Check the validation
    R = pose[:,:,:3,:3]
    T = pose[:,:,:3,3:] 

    kp_arr = kp_arr.unsqueeze(1)
    reproj_val = ((K1@R)@ (torch.inverse(K0)))@kp_arr.transpose(3,2) # TODO: ERROR

    proj_z = K1@T/Z
    reproj = reproj_val + proj_z
    reproj = reproj/reproj[:,:,2:,:]

    return reproj[:,:,:2,:]

def patch_for_kp(keypoints,ker_size,out_length,roi_patch):
    """Creates patch for key-point"""

    keypts_array = keypoints.unsqueeze(1)
    n_view = roi_patch.shape[1]
    keypts_array = keypts_array.repeat(1,n_view,1,1)
    
    xc = keypts_array[:,:,:,0]
    yc = keypts_array[:,:,:,1]

    h = torch.ones((keypts_array.shape[0],n_view,keypts_array.shape[2])).to(roi_patch.device)*ker_size  #3 #kernel_size
    w = ker_size*roi_patch[:,:,:,3]/out_length
    theta = torch.zeros((keypts_array.shape[0],n_view,keypts_array.shape[2])).to(roi_patch.device)

    
    keypoint_patch = torch.stack((xc,yc,h,w,theta),3)
    return keypoint_patch


def match_corr(embed_ref, embed_srch):
    """ Matches the two embeddings using the correlation layer. As per usual
    it expects input tensors of the form [B, C, H, W].
    Args:
        embed_ref: (torch.Tensor) The embedding of the reference image, or
            the template of reference (the average of many embeddings for
            example).
        embed_srch: (torch.Tensor) The embedding of the search image.
    Returns:
        match_map: (torch.Tensor) The correlation between
    """

    _,_, k1, k2 =  embed_ref.shape
    b, c, h, w = embed_srch.shape

    if k1==1 and k2==1:
        pad_img = (0,0)
    else:
        pad_img = (0,1)
    match_map = torch.nn.functional.conv2d(embed_srch.contiguous().view(1, b * c, h, w),embed_ref, groups=b,padding= pad_img)

    match_map = match_map.permute(1, 0, 2, 3)

    return match_map

def create_transform_matrix(roi_patch):
    """Creates a 3x3 transformation matrix for the patches"""
    transform_matrix = torch.zeros((roi_patch.shape[0], roi_patch.shape[1],roi_patch.shape[2],3,3)).to(roi_patch.device)
    transform_matrix[:,:,:,0,0] = torch.cos(roi_patch[:,:,:,4])
    transform_matrix[:,:,:,0,1] = -torch.sin(roi_patch[:,:,:,4])
    transform_matrix[:,:,:,0,2] = roi_patch[:,:,:,0]
    transform_matrix[:,:,:,1,0] = torch.sin(roi_patch[:,:,:,4])
    transform_matrix[:,:,:,1,1] = torch.cos(roi_patch[:,:,:,4])
    transform_matrix[:,:,:,1,2] = roi_patch[:,:,:,1]
    transform_matrix[:,:,:,2,2] = 1.0

    return transform_matrix

def patch_sampler(roi_patch, out_length=640, distance = 2,do_img = True ,align_corners=False):
    """Creates, scales and aligns the patch"""

    ##create a regular grid centered at xc,yc
    if out_length>1:
        width_sample = torch.linspace(-0.5, 0.5, steps=out_length)
    else:
        width_sample = torch.tensor([0.])


    height_sample = torch.linspace(-distance, distance, steps=2*distance+1)
    xv, yv = torch.meshgrid([width_sample, height_sample])
    zv = torch.ones(xv.shape)
    patch_sample = torch.stack((xv,yv,zv),2).to(roi_patch.device)


    arange_array = patch_sample.repeat(roi_patch.shape[0],roi_patch.shape[1],roi_patch.shape[2],1,1,1)

    ## scaling the x dimension to ensure unform sampling
    arange_array[:,:,:,:,:,0] = (roi_patch[:,:,:,[3]].unsqueeze(4))*arange_array[:,:,:,:,:,0]
    aras = arange_array.shape    
    arange_array = arange_array.contiguous().view(aras[0],aras[1],aras[2],aras[3]*aras[4],aras[5]).transpose(4,3)

    #create matrix transform
    transform_matrix = create_transform_matrix(roi_patch)
    #transform 
    patch_kp = transform_matrix@arange_array

    patch_kp = patch_kp.view(aras[0],aras[1],aras[2],aras[5],aras[3],aras[4])
    patch_kp = patch_kp[:,:,:,:2,:,:].transpose(5,3)
    return patch_kp, transform_matrix


def patch_for_depth_guided_range(keypoints, pose, K0, K1, img_shape, distance=2,
                                 min_depth=0.5, max_depth=10.0, align_corners=False):
    """Represents search patch for a key-point using xc,yc, h,w, theta"""
    '''
    keypoints: [B, N, 2]
    intrinsic: [B, 3, 3]
    pose: [B, 1, 4, 4]
    '''

    #get epilines
    n_view = pose.shape[1]
    pts = keypoints

    kp_arr = torch.ones((pts.shape[0],pts.shape[1],3)).to(pts.device) 
    kp_arr[:,:,:2] = pts
    kp_arr = kp_arr.unsqueeze(1) 
    # Fund,_ = get_fundamental_matrix(pose, intrinsic, intrinsic)
    Fund, _ = get_fundamental_matrix(pose, K0, K1)
    lines_epi = (Fund@(kp_arr.transpose(3,2))).transpose(3,2)

    #image shape
    height = img_shape[2] 
    width = img_shape[3]

    #default intercepts
    array_zeros = torch.zeros((pts.shape[0], n_view, pts.shape[1])).to(pts.device)
    array_ones = torch.ones((pts.shape[0], n_view, pts.shape[1])).to(pts.device)

    x2ord = array_zeros.clone().detach()
    y2ord = array_zeros.clone().detach()

    x3ord = array_zeros.clone().detach()
    y3ord = array_zeros.clone().detach()

    x0_f = array_zeros.clone().detach()
    y0_f = array_zeros.clone().detach()
    x1_f = array_zeros.clone().detach()
    y1_f = array_zeros.clone().detach()

    ##get x2,x3 and order
    x2_y2 = reproject_points(pose, keypoints, K0, K1, min_depth)
    x2 = x2_y2[:,:,0,:]
    y2 = x2_y2[:,:,1,:]
    x3_y3 = reproject_points(pose, keypoints, K0, K1, max_depth)
    x3 = x3_y3[:,:,0,:]
    y3 = x3_y3[:,:,1,:]

    # print("LCH_DEBUG:: x2_y2 value min is {}, max is {}".format(x2_y2.min(), x2_y2.max()))
    # print("LCH_DEBUG:: x3_y3 value min is {}, max is {}".format(x3_y3.min(), x3_y3.max()))


    x_ord = x3 >= x2
    x2ord[x_ord] = x2[x_ord]
    y2ord[x_ord] = y2[x_ord]
    x3ord[x_ord] = x3[x_ord]
    y3ord[x_ord] = y3[x_ord]

    cx_ord = x2>x3
    x2ord[cx_ord] = x3[cx_ord]
    y2ord[cx_ord] = y3[cx_ord]
    x3ord[cx_ord] = x2[cx_ord]
    y3ord[cx_ord] = y2[cx_ord]

    if align_corners:
        x_ord0 = (x2ord>=0) & (x2ord<width)
        x_ord1 = (x3ord>=0) & (x3ord<width)

        y_ord0 = (y2ord>=0) & (y2ord<height)
        y_ord1 = (y3ord>=0) & (y3ord<height)
    else:
        x_ord0 = (x2ord>=-0.5) & (x2ord<(width-0.5))
        x_ord1 = (x3ord>=-0.5) & (x3ord<(width-0.5))

        y_ord0 = (y2ord>=-0.5) & (y2ord<(height-0.5))
        y_ord1 = (y3ord>=-0.5) & (y3ord<(height-0.5))

    all_range = x_ord0 & x_ord1 & y_ord0 & y_ord1

    x0_f[all_range]  = x2ord[all_range]
    y0_f[all_range]  = y2ord[all_range]

    x1_f[all_range]  = x3ord[all_range]
    y1_f[all_range]  = y3ord[all_range]

    cond_null = ~all_range
    x0_f[cond_null] = array_zeros.clone().detach()[cond_null]
    y0_f[cond_null] = array_zeros.clone().detach()[cond_null]
    x1_f[cond_null] = array_zeros.clone().detach()[cond_null]
    y1_f[cond_null] = array_zeros.clone().detach()[cond_null]

    ## find box representation using #xc, yc, h,w, theta # x0_f, x1_f is are the reprojected location with the min depth and max depth respectively.
    xc = (x0_f+x1_f)/2. 
    yc = (y0_f+y1_f)/2. 
    h = torch.ones((pts.shape[0],n_view,pts.shape[1])).to(pts.device)*max(2*distance,1)
    w = torch.sqrt((x1_f-x0_f)**2+(y1_f-y0_f)**2)  # radian?
    # print("LCH_DEBUG:: w value min is {}, max is {}".format(w.min(), w.max()))
    theta = torch.atan2(-lines_epi[:,:,:,0], lines_epi[:,:,:,1])

    if torch.sum(torch.isnan(theta)):
        import pdb; pdb.set_trace()
    roi_patch = torch.stack((xc,yc,h,w,theta),3)
    
    return roi_patch


def sample_descriptors_epi(keypoints, descriptors, s, normalize =  True, align_corner = False):
    """Samples descriptors at point locations"""

    b, c, h, w = descriptors.shape


    keypoints = keypoints - s / 2 + 0.5 
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],device=keypoints.device)[None]

    keypoints = keypoints*2 - 1  
    if len(keypoints.shape) == 4:
        descriptors = torch.nn.functional.grid_sample(descriptors, keypoints.view(b, keypoints.shape[1], keypoints.shape[2], 2), mode='bilinear',align_corners =align_corner) ##pythorch 1.3+
    elif len(keypoints.shape) == 3:
        descriptors = torch.nn.functional.grid_sample(descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear',align_corners =align_corner) ##pythorch 1.3+

    if normalize:
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

    return descriptors



def vec_to_skew_symmetric(v):
    """Creates skew-symmetric matrix"""
    zero = torch.zeros_like(v[:, 0])
    M = torch.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], dim=1)
    return M.reshape(-1, 3, 3)


def get_fundamental_matrix(T_10, K0, K1):
    """Generates fundamental matrix"""

    ##Expects BX3x3 matrix 
    k0 = torch.inverse(K0) 
    k1 = torch.inverse(K1).transpose(1, 2)

    k0 = k0.unsqueeze(1)
    k1 = k1.unsqueeze(1)

    T_10 = T_10.view(-1,4,4)
    t_skew = vec_to_skew_symmetric(T_10[:, :3, 3])
    E = t_skew @ T_10[:, :3, :3] ##Essential matrix
    E = E.view(k0.shape[0],-1,3,3)

    Fu = (k1@E)@k0 ##Fundamental matrix
    F_norm = Fu[:,:,2:,2:]
    F_norm[F_norm==0.] = 1.
    Fu = Fu/F_norm  ##normalize it
    return Fu, E



class TriangulationNet(nn.Module):
    """Triangulation module"""
    default_config = {
        'min_depth': 0.0,
        'max_depth': 30.0,
        'align_corners': False,
        'depth_range': True,
        'arg_max_weight': 1.0,
        'dist_orthogonal': 1,
        'siamese': False,
        'densification': False
    }

    def __init__(self, config, image_shape=(640, 480)):
        super().__init__()
        self.config = {**self.default_config, **config}
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.image_shape = image_shape

    def triangulate(self, keypoints, kp_matched, confidence, pose, K0, K1):
        '''
        keypoints: [B, N, 2]
        kp_matched: [B, N, 2]
        pose0: [N, 4, 4]
        img_shape: [B, C, H, W]
        confidence: [B, 1, N]
        B = 1
        return: sparse_depth_learnt: [1, 1, H, W]
        '''

        confidence = confidence.unsqueeze(1)
        self_confidence = torch.ones((confidence.shape[0], 1, confidence.shape[2])).to(confidence.device)
        confidence = torch.cat((self_confidence, confidence), 1)
        confidence = confidence.transpose(2, 1)
        multiview_matches = torch.cat((keypoints.unsqueeze(1), kp_matched.unsqueeze(1)), 1)

        ####  Triangulation
        pose_tiled = pose[:, :3, :]

        projection_mat = []
        projection_ref = []
        proj_identity = torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]], device=confidence.device)

        for batch_idx in range(pose_tiled.size(0)):
            proj_ref_idx = torch.mm(K0[batch_idx],proj_identity).unsqueeze(0)
            projection_ref.append(proj_ref_idx)

            # print("LCH_DEBUG:: K1 shape is {}, pose_tiled shape is {}".format(K1.shape, pose_tiled.shape))
            proj_mat_view = torch.mm(K1[batch_idx], pose_tiled[batch_idx]).unsqueeze(0)
            projection_mat.append(proj_mat_view)

        projection_mat = torch.cat(projection_mat, 0).unsqueeze(1)
        projection_ref = torch.cat(projection_ref, 0).unsqueeze(1)

        proj_matrices = torch.cat([projection_ref, projection_mat],1)

        keypoints_3d = triangulate_batch_of_points(proj_matrices, multiview_matches, confidence)

        keypoints_3d = torch.stack(keypoints_3d, 0)
        if torch.sum(torch.isinf(keypoints_3d))>0:
            keypoints_3d = torch.clamp(keypoints_3d, min=-1000.0, max=1000.0)

        kp3d_val = keypoints_3d[:,:,2].view(-1,1).t()
        kp3d_val = torch.clamp(kp3d_val, min=0.0, max=self.config['max_depth'])
        kp3d_filter = (kp3d_val>self.config['min_depth']) & (kp3d_val<self.config['max_depth'])
        kp3d_val = (kp3d_val * kp3d_filter.float()).squeeze(0)
        keypoints_3d = keypoints_3d.view(-1, 3)
        return kp3d_val, keypoints_3d

    def generate_sparse_depth(self, kp3d_val, keypoints, img_shape):
        # Impute learnt sparse depth into a sparse image
        sparse_depth_learnt = torch.zeros((1, img_shape[-2], img_shape[-1])).to(keypoints.device)
        keypoints_index = keypoints.long()
        bselect = torch.arange(keypoints.shape[0], dtype=torch.long)
        bselect = bselect.unsqueeze(1).unsqueeze(1)
        bselect = bselect.repeat(1, keypoints_index.shape[1], 1).to(keypoints.device)
        keypoints_indexchunk = torch.cat((bselect, keypoints_index[:,:,[1]], keypoints_index[:,:,[0]]),2)
        keypoints_indexchunk = keypoints_indexchunk.view(-1,3).t()
        sparse_depth_learnt[keypoints_indexchunk.chunk(chunks=3, dim=0)] = kp3d_val
        return sparse_depth_learnt

    def forward(self, data):
        pose0 = data['T_0to1']  # [N, 4, 4]
        K0 = data['K0']  # [N, 3, 3]
        K1 = data['K1']  # [N, 3, 3]
        confidence = data['mconf']  # [N,]
        keypoints = data['mkpts0_f']  # [N, 2]
        kp_matched = data['mkpts1_f']  # [N, 2]
        img_shape = data['image0'].shape  # [B, C, H, W]

        if len(K0.shape) == 2:
            K0 = K0.unsqueeze(0)
        if len(K1.shape) == 2:
            K1 = K1.unsqueeze(0)

        kp_batch_list = []
        kp_matched_batch_list = []
        conf_batch_list = []
        for b in range(img_shape[0]):
            kp_batch_list.append(keypoints[data["m_bids"]==b].unsqueeze(0))
            kp_matched_batch_list.append(kp_matched[data["m_bids"]==b].unsqueeze(0))
            conf_batch_list.append(confidence[data["m_bids"]==b].unsqueeze(0))

        if self.config['densification']:
            sparse_depth_learned_list = []
            sparse_depth_value_list = []
            for i in range(len(kp_batch_list)):
                kps = kp_batch_list[i]  # [1, N, 2]
                kps_m = kp_matched_batch_list[i]  # [1, N, 2]
                conf = conf_batch_list[i]  # [1, N]
                k0 = K0[[i], :, :]
                k1 = K1[[i], :, :]
                T = pose0[[i], :, :]

                # TODO: Triangulate before scale to 640 x 640, generate sparse_depth_learnt after Triangulate
                sparse_depth_learnt_local = torch.zeros((img_shape[0], img_shape[-2], img_shape[-1])).to(kps.device)
                sparse_depth_value = torch.zeros((kps.shape[1]), device=kps.device)
                if kps.shape[1] != 0:
                    sparse_depth_value, _ = self.triangulate(kps, kps_m, conf, T, k0, k1)
                    if 'scale0' in data:
                        scale0 = data['scale0'][[i], :]
                        kps = kps / scale0
                    sparse_depth_learnt_local = self.generate_sparse_depth(sparse_depth_value, kps, img_shape)
                sparse_depth_learned_list.append(sparse_depth_learnt_local)
                sparse_depth_value_list.append(sparse_depth_value)
            sparse_depth_learnt0 = torch.cat(sparse_depth_learned_list, dim=0)
            sparse_depth_value0 = torch.cat(sparse_depth_value_list, dim=0)
            data.update({'depth0_sparse': sparse_depth_learnt0, 'depth0_sparse_value': sparse_depth_value0})

            # Training
            if self.config['siamese']:
                pose1 = data['T_1to0']  # [N, 4, 4]
                sparse_depth_learned_list = []
                sparse_depth_value_list = []

                for i in range(len(kp_batch_list)):
                    kps = kp_batch_list[i]  # [1, N, 2]
                    kps_m = kp_matched_batch_list[i]  # [1, N, 2]
                    conf = conf_batch_list[i]  # [1, N]
                    k0 = K0[[i], :, :]
                    k1 = K1[[i], :, :]
                    T = pose1[[i], :, :]

                    sparse_depth_learnt_local = torch.zeros((img_shape[0], img_shape[-2], img_shape[-1]), device=kps.device)
                    sparse_depth_value = torch.zeros((kps.shape[1]), device=kps.device)
                    if kps.shape[1] != 0:
                        sparse_depth_value, _ = self.triangulate(kps_m, kps, conf, T, k1, k0)
                        if 'scale1' in data:
                            scale1 = data['scale1'][[i], :]
                            kps_m = kps_m / scale1
                        sparse_depth_learnt_local = self.generate_sparse_depth(sparse_depth_value, kps_m, img_shape)
                    sparse_depth_learned_list.append(sparse_depth_learnt_local)
                    sparse_depth_value_list.append(sparse_depth_value)

                sparse_depth_learnt1 = torch.cat(sparse_depth_learned_list, dim=0)
                sparse_depth_value1 = torch.cat(sparse_depth_value_list, dim=0)
                data.update({'depth1_sparse': sparse_depth_learnt1, 'depth1_sparse_value': sparse_depth_value1})
        else:
            sparse_depth_value_list = []
            kps_3d_list = []
            for i in range(len(kp_batch_list)):
                kps = kp_batch_list[i]  # [1, N, 2]
                kps_m = kp_matched_batch_list[i]  # [1, N, 2]
                conf = conf_batch_list[i]  # [1, N]
                k0 = K0[[i], :, :]
                k1 = K1[[i], :, :]
                T = pose0[[i], :, :]
                sparse_depth_value = torch.zeros((kps.shape[1]), device=kps.device)
                kps_3d = torch.zeros(kps.shape[1], 3, device=kps.device)
                if kps.shape[1] != 0:
                    sparse_depth_value, kps_3d = self.triangulate(kps, kps_m, conf, T, k0, k1)
                sparse_depth_value_list.append(sparse_depth_value)
                kps_3d_list.append(kps_3d)
            sparse_depth_value0 = torch.cat(sparse_depth_value_list, dim=0)
            kps0_3d = torch.cat(kps_3d_list, dim=0)
            data.update({'depth0_sparse_value': sparse_depth_value0, 'mkpts0_f_3d': kps0_3d})

            if self.config['siamese']:
                pose1 = data['T_1to0']  # [N, 4, 4]
                sparse_depth_value_list = []
                kps_3d_list = []
                for i in range(len(kp_batch_list)):
                    kps = kp_batch_list[i]  # [1, N, 2]
                    kps_m = kp_matched_batch_list[i]  # [1, N, 2]
                    conf = conf_batch_list[i]  # [1, N]
                    k0 = K0[[i], :, :]
                    k1 = K1[[i], :, :]
                    T = pose1[[i], :, :]

                    kps_3d = torch.zeros(kps.shape[1], 3, device=kps.device)
                    sparse_depth_value = torch.zeros((kps.shape[1]), device=kps.device)
                    if kps.shape[1] != 0:
                        sparse_depth_value, kps_3d = self.triangulate(kps_m, kps, conf, T, k1, k0)
                    sparse_depth_value_list.append(sparse_depth_value)
                    kps_3d_list.append(kps_3d)
                sparse_depth_value1 = torch.cat(sparse_depth_value_list, dim=0)
                kps1_3d = torch.cat(kps_3d_list, dim=0)
                data.update({'depth1_sparse_value': sparse_depth_value1, 'mkpts1_f_3d': kps1_3d})