'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
# from visu_util import visualize

def chamfer_dist_loss(a, b):
    # x, y = a.transpose(2, 1).contiguous(), b.transpose(2, 1).contiguous()
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    loss = (torch.mean(P.min(1)[0])) + (torch.mean(P.min(2)[0]))
    return loss

def ghmm_params(gamma1, pts1, pts1_normal, gamma2, pts2, pts2_normal):
    '''
    Inputs:
          gamma: B x N x J
          pts: B x N x 3
          pts_normal: B x N x 3

    Outputs:
          pi: B x J
          mu: B x J x 3
          sigma: B

          mu_normal: B x J x 3
          kappa: B
    '''
    B = gamma1.shape[0]
    N = gamma1.shape[1]
    J = gamma1.shape[2]
    
    # pi: B x 1 x J
    # pi1, pi2 = gamma1.mean(dim=1), gamma2.mean(dim=1)
    # Npi = pi1 * gamma1.shape[1] + pi2 * gamma2.shape[1]
    d = gamma1.sum(dim=1, keepdim=True) + gamma2.sum(dim=1, keepdim=True)
    
    '''Positional Vectors Related''' 
    # mu: B x J x 3
    mu = (gamma1.transpose(1, 2) @ pts1 + gamma2.transpose(1, 2) @ pts2) / d.transpose(1, 2)
    
    # B x N x J x 3
    diff_pts1 = pts1.unsqueeze(2) - mu.unsqueeze(1)
    diff_pts2 = pts2.unsqueeze(2) - mu.unsqueeze(1)
    
    sigma_numerator = ((diff_pts1.unsqueeze(3) @ diff_pts1.unsqueeze(4)).squeeze() * gamma1)+ \
        ((diff_pts2.unsqueeze(3) @ diff_pts2.unsqueeze(4)).squeeze() * gamma2)
    sigma = (sigma_numerator.sum(dim=1, keepdim=True) / d / 3).squeeze(1)
    
    '''Normal Vectors Related'''
    # pts_normal_norm: B x J 
    gamma_pts1_normal_product =  gamma1.transpose(1, 2) @ pts1_normal # B x J x 3
    gamma_pts2_normal_product =  gamma2.transpose(1, 2) @ pts2_normal # B x J x 3  
    pts_normal_norm = torch.norm((gamma_pts1_normal_product + gamma_pts2_normal_product), dim=2) 
    # mu_normal: B x J x 3
    mu_normal = (gamma_pts1_normal_product + gamma_pts2_normal_product) / pts_normal_norm.unsqueeze(2)
    
    # pts1_normal_expand:B x N x J x 3
    pts1_normal_unsqueeze = pts1_normal.unsqueeze(2) # B x N x 1 x 3
    pts1_normal_expand    = pts1_normal_unsqueeze.repeat(1,1,J,1)
    pts2_normal_unsqueeze = pts2_normal.unsqueeze(2) # B x N x 1 x 3
    pts2_normal_expand    = pts2_normal_unsqueeze.repeat(1,1,J,1)
    # mu_normal_expand:B x N x J x 3
    mu_normal_unsqueeze  = mu_normal.unsqueeze(1) # B x 1 x J x 3
    mu_normal_expand     = mu_normal_unsqueeze.repeat(1,N,1,1)
    # kappa:B x J
    kappa_numerator = (((pts1_normal_expand.unsqueeze(3) @ mu_normal_expand.unsqueeze(4)).squeeze() * gamma1))\
        + (((pts2_normal_expand.unsqueeze(3) @ mu_normal_expand.unsqueeze(4)).squeeze() * gamma2))
    kappa = (kappa_numerator.sum(dim=1, keepdim=True) / d).squeeze(1)
    
    return mu, sigma, mu_normal, kappa
    # return mu, sigma


def ghmm_register(gamma, pts, pts_normal, mu, mu_normal, sigma, kappa):
    """_summary_

    Args:
        gamma (_type_): B x N x J
        pts (_type_): B x N x 3
        pts_normal (_type_): B x N x 3
        mu (_type_): B x J x 3
        mu_normal (_type_): B x J x 3
        sigma (_type_): B
        kappa (_type_): B

    Returns:
        _type_: _description_
    """    
    W = (gamma.transpose(2, 1) @ pts) / sigma.unsqueeze(-1)
    mean_pts = W.sum(dim=1, keepdim=True)
    
    mean_mu = (mu.transpose(1, 2) @ (gamma.sum(dim=1)/sigma).unsqueeze(-1)).transpose(1, 2)
        
    sow = (gamma.sum(dim=1, keepdim=True) / sigma.unsqueeze(1)).sum(dim=2)
    H1 = mu.transpose(1, 2) @ W - mean_mu.transpose(1, 2) @ mean_pts / sow.unsqueeze(2)
    
    # H2 matrix
    # pts_normal_s_expand: B x M x 3 x 1
    pts_normal_expand = pts_normal.unsqueeze(3)
    # mu_normal_t_expand: B x J x 1 x 3
    mu_normal_expand  = mu_normal.unsqueeze(2)
    
    H2_numerator = (((pts_normal_expand.unsqueeze(2) @ mu_normal_expand.unsqueeze(1)) * \
                    (gamma.unsqueeze(3).unsqueeze(4))).sum(dim=1))
    H2 = (H2_numerator / kappa.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3,3)).sum(dim=1)
    
    H = H1 + H2
    
    # Compute the rotation matrix R1 with SVD 
    U, _, V = torch.svd(H.cpu())
    U = U.cuda()
    V = V.cuda()
    S = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
    S[:, 2, 2] = torch.det(V @ U.transpose(1, 2))
    # R: B x 3 x 3
    R = V @ S @ U.transpose(1, 2)
    t = (mean_mu.transpose(1, 2) - R @ mean_pts.transpose(1, 2)) / sow.unsqueeze(2)
    return R, t.transpose(1, 2)


def rotation_error(R, R_gt):
    cos_theta = (torch.einsum('bij,bij->b', R, R_gt) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    return torch.acos(cos_theta) * 180 / math.pi


def translation_error(t, t_gt):
    # t = T[:, :3, 3] * scale.unsqueeze(1) - (T[:, :3, :3] @ centroid.reshape(-1, 3, 1)).squeeze(-1)
    # t_gt = T_gt[:, :3, 3] * scale.unsqueeze(1) - (T_gt[:, :3, :3] @ centroid.reshape(-1, 3, 1)).squeeze(-1)
    # return torch.norm(T[:, :3, 3] - T_gt[:, :3, 3], dim=1)
    return torch.norm(t - t_gt, dim=1)


def rmse(pts, T, T_gt):
    pts_pred = pts @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
    pts_gt = pts @ T_gt[:, :3, :3].transpose(1, 2) + T_gt[:, :3, 3].unsqueeze(1)
    return torch.norm(pts_pred - pts_gt, dim=2).mean(dim=1)


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    message = torch.einsum('bhnm,bdhm->bdhn', prob, value)
    return message, prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """

    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        self.norm = nn.InstanceNorm1d(d_model)
        # self.norm2 = nn.InstanceNorm1d(d_model)

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        x = self.merge(x.contiguous().view(batch_dim, self.dim * self.num_heads, -1))
        x = self.norm(x)
        # x = self.norm2(x)

        return x


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim * 2, feature_dim * 2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class Transformer(nn.Module):
    def __init__(self, num_head: int, feature_dim: int):
        super().__init__()

        self.attention_layer = AttentionalPropagation(feature_dim, num_head)

    def forward(self, desc0, desc1):
        desc0_ca = self.attention_layer(desc0, desc1)
        desc1_ca = self.attention_layer(desc1, desc0)

        return desc0_ca, desc1_ca
    

class Conv1dBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(Conv1dBNReLU, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True))

class FCBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes):
        super(FCBNReLU, self).__init__(
            nn.Linear(in_planes, out_planes, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True))


class TNet(nn.Module):
    def __init__(self):
        super(TNet, self).__init__()
        self.encoder = nn.Sequential(
            Conv1dBNReLU(6, 64),
            Conv1dBNReLU(64, 128),
            Conv1dBNReLU(128, 256))
        self.decoder = nn.Sequential(
            FCBNReLU(256, 128),
            FCBNReLU(128, 64),
            nn.Linear(64, 6))

    @staticmethod
    def f2R(f):
        r1 = F.normalize(f[:, :3])
        proj = (r1.unsqueeze(1) @ f[:, 3:].unsqueeze(2)).squeeze(2)
        r2 = F.normalize(f[:, 3:] - proj * r1)
        r3 = r1.cross(r2)
        return torch.stack([r1, r2, r3], dim=2)

    def forward(self, pts):
        f = self.encoder(pts)
        f, _ = f.max(dim=2)
        f = self.decoder(f)
        R = self.f2R(f)
        tnet_position = R @ pts[:, 0:3, :]
        tnet_normal   = R @ pts[:, 3:6, :]
        output = torch.cat((tnet_position, tnet_normal), 1)
        return output # R @ pts # for positional vectors only

class PointNet(nn.Module):
    def __init__(self, args):
        super(PointNet, self).__init__()
        self.use_tnet = args.use_tnet
        self.tnet = TNet() if self.use_tnet else None
        d_input = args.k * 4 if args.use_rri else 6
        self.encoder = nn.Sequential(
            Conv1dBNReLU(d_input, 64),
            Conv1dBNReLU(64, 128),
            Conv1dBNReLU(128, 256),
            Conv1dBNReLU(256, args.d_model))
        self.decoder = nn.Sequential(
            Conv1dBNReLU(args.d_model * 2, 512),
            Conv1dBNReLU(512, 256),
            Conv1dBNReLU(256, 128),
            nn.Conv1d(128, args.n_clusters, kernel_size=1))

    def forward(self, pts):
        pts = self.tnet(pts) if self.use_tnet else pts      # [B, 6, 1024]
        f_loc = self.encoder(pts)
        f_glob, _ = f_loc.max(dim=2)
        f_glob = f_glob.unsqueeze(2).expand_as(f_loc)
        y = self.decoder(torch.cat([f_loc, f_glob], dim=1))
        return y.transpose(1, 2)        # return: [B, GMM, 1024]



def parsing_predicts(R_x, t_x, R_y, t_y):

    R_xy = R_x.transpose(2, 1) @ R_y
    R_yx = R_y.transpose(2, 1) @ R_x
    t_xy = (t_x - t_y) @ R_y
    t_yx = (t_y - t_x) @ R_x

    bot_row = torch.Tensor([[[0, 0, 0, 1]]]).repeat(R_xy.shape[0], 1, 1).to(R_xy.device)
    pred_xy = torch.cat([torch.cat([R_xy.transpose(2, 1), t_xy.transpose(1, 2)], dim=2), bot_row], dim=1)
    pred_yx = torch.cat([torch.cat([R_yx.transpose(2, 1), t_yx.transpose(1, 2)], dim=2), bot_row], dim=1)

    return pred_xy, pred_yx

class GHMM(nn.Module):
    def __init__(self, args):
        super(GHMM, self).__init__()
        self.use_tnet = args.use_tnet
        self.tnet = TNet() if self.use_tnet else None
        self.mlp1 = Conv1dBNReLU(6, 64)
        self.mlp2 = Conv1dBNReLU(64, 64)
        self.Transformer1 = Transformer(1, 64)
        
        self.mlp3 = Conv1dBNReLU(128, 128)
        self.mlp4 = Conv1dBNReLU(128, 256)
        self.Transformer2 = Transformer(4, 256)
        
        self.mlp5 = Conv1dBNReLU(512, 512)
        
        self.Clustering = nn.Sequential(
            Conv1dBNReLU(1024, 1024),
            Conv1dBNReLU(1024, 512),
            Conv1dBNReLU(512, 128),
            Conv1dBNReLU(128, 64),
            nn.Conv1d(64, args.n_clusters, kernel_size=1)
        )

    def forward(self, pts1, pts2, pts1_normal, pts2_normal, T_gt, scale):
        self.pts1 = pts1                # B x M x 3
        # self.pts2 = pts2                # B x N x 3

        self.pts1_normal = pts1_normal  # B x M x 3
        self.pts2_normal = pts2_normal  # B x N x 3

        ## correspondence module
        feats_pts1 = (pts1 - pts1.mean(dim=1, keepdim=True)).transpose(1, 2) # B x 3 x M
        feats_pts2 = (pts2 - pts2.mean(dim=1, keepdim=True)).transpose(1, 2) # B x 3 x N
        
        feats_normal1 = pts1_normal.transpose(1, 2) # B x 3 x M
        feats_normal2 = pts2_normal.transpose(1, 2) # B x 3 x N

        pts1_f0 = torch.cat((feats_pts1, feats_normal1), 1) # B x 6 x M
        pts2_f0 = torch.cat((feats_pts2, feats_normal2), 1) # B x 6 x N
        
        pts1_f0 = self.tnet(pts1_f0)
        pts2_f0 = self.tnet(pts2_f0)
        
        pts1_f1 = self.mlp1(pts1_f0)
        pts2_f1 = self.mlp1(pts2_f0)
        
        pts1_f2 = self.mlp2(pts1_f1)
        pts2_f2 = self.mlp2(pts2_f1)
        
        pts1_f2_ca, pts2_f2_ca = self.Transformer1(pts1_f2, pts2_f2)
        
        pts1_f_lg = torch.cat([pts1_f2, pts1_f2_ca], dim=1)  # 128
        pts2_f_lg = torch.cat([pts2_f2, pts2_f2_ca], dim=1)
        
        pts1_f3 = self.mlp3(pts1_f_lg)  # 128
        pts2_f3 = self.mlp3(pts2_f_lg)

        pts1_f4 = self.mlp4(pts1_f3)  # 256
        pts2_f4 = self.mlp4(pts2_f3)
        
        # Global info interaction
        pts1_f4_ca, pts2_f4_ca = self.Transformer2(pts1_f4, pts2_f4)
        
        pts1_f_lg = torch.cat([pts1_f4, pts1_f4_ca], dim=1)  # 512
        pts2_f_lg = torch.cat([pts2_f4, pts2_f4_ca], dim=1)
        
        pts1_f5 = self.mlp5(pts1_f_lg)
        pts2_f5 = self.mlp5(pts2_f_lg)
        
        pts1_f_final = pts1_f5
        pts2_f_final = pts2_f5
        
        pts1_final_g = pts1_f_final.max(dim=2, keepdim=True)[0].repeat(1, 1, pts1.shape[1])
        pts2_final_g = pts2_f_final.max(dim=2, keepdim=True)[0].repeat(1, 1, pts2.shape[1])

        self.gamma1 = F.softmax(self.Clustering(torch.cat([pts1_f_final, pts1_final_g], dim=1)).transpose(2, 1), dim=2)
        self.gamma2 = F.softmax(self.Clustering(torch.cat([pts2_f_final, pts2_final_g], dim=1)).transpose(2, 1), dim=2)
        
        # Compute model paramters of GHMM
        self.mu, self.sigma, self.mu_normal, self.kappa = ghmm_params(self.gamma1, pts1, pts1_normal, self.gamma2, pts2, pts2_normal)    
        # self.mu, self.sigma = ghmm_params(self.gamma1, pts1, pts1_normal, self.gamma2, pts2, pts2_normal)                 
    
        # Compute the transformation
        R1_w, t1_w_unit = ghmm_register(self.gamma1, pts1, pts1_normal, self.mu, self.mu_normal, self.sigma, self.kappa)
        R2_w, t2_w_unit = ghmm_register(self.gamma2, pts2, pts2_normal, self.mu, self.mu_normal, self.sigma, self.kappa)
        
        T_12_unit, T_21_unit = parsing_predicts(R1_w, t1_w_unit, R2_w, t2_w_unit)

        T_gt_mm = T_gt.clone()
        T_gt_mm[:, :3, 3] = T_gt_mm[:, :3, 3] * scale.unsqueeze(1)
        
        px_w = pts1 @ R1_w.transpose(2, 1) + t1_w_unit
        py_w = pts2 @ R2_w.transpose(2, 1) + t2_w_unit
        eye = torch.eye(4, dtype=torch.float, device=T_12_unit.device).unsqueeze(0).repeat(T_12_unit.shape[0], 1, 1)
        # self.mse1 = F.mse_loss(T_12_unit @ torch.inverse(T_gt_unit), eye)
        # self.mse2 = F.mse_loss(T_21_unit @ T_gt_unit, eye)
        # loss = 3 * F.mse_loss(T_12_unit @ T_21_unit, eye) + chamfer_dist_loss(px_w, py_w)  # + self.mse1 + self.mse2
        loss = chamfer_dist_loss(px_w, py_w) + F.mse_loss(pts1_normal @ T_12_unit[:, :3, :3].transpose(1, 2), pts2_normal)

        # T_gt_unit = T_gt_mm.clone()
        # # T_gt_unit[:, :3, 3] = (T_gt_unit[:, :3, 3] - centroid + (T_gt_unit[:, :3, :3] @ centroid.reshape(-1, 3, 1)).squeeze(-1)) / scale.unsqueeze(1)
        # T_gt_unit[:, :3, 3] = (T_gt_unit[:, :3, 3]) / scale.unsqueeze(1)

        # eye = torch.eye(4).expand_as(T_gt_unit).to(T_gt_unit.device)
        # self.mse1 = F.mse_loss(T_12_unit @ torch.inverse(T_gt_unit), eye)
        # self.mse2 = F.mse_loss(T_21_unit @ T_gt_unit, eye)
        # loss = self.mse1 + self.mse2
        # loss = self.mse1 # forward loss only

        T_12_mm = T_12_unit.clone()
        T_12_mm[:, :3, 3] = T_12_mm[:, :3, 3] * scale.unsqueeze(1) #- (T_12_mm[:, :3, :3] @ centroid.reshape(-1, 3, 1)).squeeze(-1) + centroid
        T_21_mm = T_21_unit.clone()
        T_21_mm[:, :3, 3] = T_21_mm[:, :3, 3] * scale.unsqueeze(1) #- (T_21_mm[:, :3, :3] @ centroid.reshape(-1, 3, 1)).squeeze(-1) + centroid
        
        
        self.r_err = rotation_error(T_12_mm[:, :3, :3], T_gt_mm[:, :3, :3])
        self.t_err = translation_error(T_12_mm[:, :3, 3], T_gt_mm[:, :3, 3])
        pts1_mm = pts1 * scale.unsqueeze(-1).unsqueeze(-1)
        self.rmse = rmse(pts1_mm[:, :100], T_12_mm, T_gt_mm)

        self.T_12_mm = T_12_mm
        
        # self.initial_r = rotation_error(torch.eye(3).unsqueeze(0).to(T_gt.device), T_gt[:, :3, :3])
        # self.initial_t = translation_error(torch.zeros(3).unsqueeze(0).to(T_gt.device), T_gt[:, :3, 3])

        return loss, self.r_err, self.t_err, self.rmse
        # return loss, self.r_err, self.t_err, self.rmse, self.T_12, self.initial_r, self.initial_t
