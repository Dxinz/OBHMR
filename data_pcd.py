'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import numpy as np
import os
from torch.utils.data import Dataset
import open3d as o3d
import glob
import math

import torch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import minkowski


def jitter_pcd(pcd, sigma=0.01, clip=0.05):
    pcd += np.clip(sigma * np.random.randn(*pcd.shape), -1 * clip, clip)
    return pcd


def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    # angle = 30 / 180 * np.pi
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    # R = np.eye(3) + np.sin(math.pi * 2 / 3) * A + (1 - np.cos(math.pi * 2 / 3)) * np.dot(A, A)
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    # t *= np.random.rand() * max_dist
    t *= 1.0 * max_dist
    # t *= 0.3
    return np.expand_dims(t, 1)


def get_pcd_data(path):

    all_data = []

    for stlfile_name in glob.glob(os.path.join(path, '*.pcd')):#glob遍历完指定路径下的所有PCD文件
        pcd_origin = o3d.io.read_point_cloud(stlfile_name)
        pcd_origin_points = np.asarray(pcd_origin.points)
        pcd_origin_normals = np.asarray(pcd_origin.normals)
        if pcd_origin_points.shape[0] != 0 and pcd_origin_normals.shape[0] != 0:
            all_data.append(np.concatenate([pcd_origin_points, pcd_origin_normals], axis=-1))
        else:
            print(stlfile_name)
    return all_data


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)

def Randomcrop(p_keep, points, points_normal):
    rand_xyz = uniform_2_sphere()
    centroid = np.mean(points[:, :3], axis=0)
    points_centered = points[:, :3] - centroid

    dist_from_plane = np.dot(points_centered, rand_xyz)
    if p_keep == 0.5:
        mask = dist_from_plane > 0
    else:
        mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)

    return points[mask, :], points_normal[mask, :]

def farthest_subsample_points(num_subsampled_points,pointcloud1,points_normal): #input n*3
    # (num_points, 3)
    pointcloud1 = pointcloud1
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1[:, :3])
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    gt_mask = torch.zeros(num_points).scatter_(0, torch.tensor(idx1), 1)

    return pointcloud1[idx1, :],points_normal[idx1, :], gt_mask

class TrainData(Dataset):
    def __init__(self, path, args):
        super(TrainData, self).__init__()
        self.points = get_pcd_data(path)  # B x M x 6
        self.n_points = args.n_points
        # self.sample = args.sample
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        # print("max_trans:", self.max_trans)
        self.noisy = not args.clean
        self.partial_rate = args.partial

    def __getitem__(self, index):
        # rand_idxs = np.random.choice(self.points.shape[1], self.n_points, replace=False)
        if self.points[index].shape[0] < self.n_points:
            rand_idxs = np.random.choice(self.points[index].shape[0], self.n_points, replace=True)
        else:
            rand_idxs = np.random.choice(self.points[index].shape[0], self.n_points, replace=False)

        pcd = self.points[index][rand_idxs,:3]
        # print("pcd:",pcd.shape)
        centroid = np.mean(pcd, axis=0)
        pcd = pcd - centroid
        scale = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)))
        # print("scale:",scale)
        pcd = pcd / scale
        pcd_normal = self.points[index][rand_idxs,3:6]
        # print("scale:",scale)
        max_angle = self.max_angle
        max_trans = self.max_trans/ scale
        #print("maxtrans:", max_trans)
        transform = random_pose(max_angle, max_trans / 2)
        pose1 = random_pose(np.pi, max_trans)
        pose2 = transform @ pose1

        if self.partial_rate != 1.0:
            # pcd1, pcd1_normal = Randomcrop(self.partial_rate, pcd, pcd_normal)
            n_points =  int(self.partial_rate*self.n_points)
            pcd1, pcd1_normal,gt_mask = farthest_subsample_points(n_points,pcd,pcd_normal)
        else:
            pcd1, pcd1_normal = pcd, pcd_normal

        pcd1 = pcd1 @ pose1[:3, :3].T + pose1[:3, 3]
        pcd1_normal = pcd1_normal @ pose1[:3, :3].T

        pcd2 = pcd @ pose2[:3, :3].T + pose2[:3, 3]
        pcd2_normal = pcd_normal @ pose2[:3, :3].T
        
        if self.noisy:
            pcd1 = jitter_pcd(pcd1)
            pcd2 = jitter_pcd(pcd2)
        
        sample = {'points': pcd, 'points_normal': pcd_normal}
        sample['pcd2'] = pcd2
        sample['pcd2_normal'] = pcd2_normal
        sample['transform_gt'] = transform
        sample['scale'] = scale
        sample['pcd1'] = pcd1
        sample['pcd1_normal'] = pcd1_normal

        if self.partial_rate != 1.0:
            sample['gt_mask'] = gt_mask


        #     # inds = np.random.choice(pcd1.shape[0], self.n_points, replace=True)
        #     # sample['pcd1'] = pcd1[inds, :3]
        #     # sample['pcd1_normal'] = pcd1_normal[inds, :3]
        #     # #
        #     res_num = self.n_points - pcd1.shape[0]
        #     residual = np.zeros((res_num, 3))
        #     sample['pcd1'] = np.concatenate([pcd1, residual], axis=0)
        #     sample['pcd1_normal'] = np.concatenate([pcd1_normal, residual], axis=0)
        #     # print(pcd2.shape)
        #     # sample['mask_gt'] = gt_mask
        # else:
        #     sample['pcd1'] = pcd1
        #     sample['pcd1_normal'] = pcd1_normal
        # return pcd1.astype('float32'), pcd2.astype('float32'), pcd1_normal.astype('float32'), pcd2_normal.astype('float32'), transform.astype('float32'), \
        #     centroid.astype('float32'), scale.astype('float32')
        # return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32'), scale.astype('float32')
        return sample
    
    def __len__(self):
        return len(self.points)


class TrainData2(Dataset):
    def __init__(self, path, args):
        super(TrainData2, self).__init__()
        self.points = get_pcd_data(path)  # B x M x 6
        self.n_points = args.n_points
        # self.sample = args.sample
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        # print("max_trans:", self.max_trans)
        self.noisy = not args.clean
        self.partial_rate = args.partial

    def __getitem__(self, index):
        # rand_idxs = np.random.choice(self.points.shape[1], self.n_points, replace=False)
        if self.points[index].shape[0] < self.n_points:
            rand_idxs = np.random.choice(self.points[index].shape[0], self.n_points, replace=True)
        else:
            rand_idxs = np.random.choice(self.points[index].shape[0], self.n_points, replace=False)

        pcd = self.points[index][rand_idxs, :3]
        # print("pcd:",pcd.shape)
        centroid = np.mean(pcd, axis=0)
        pcd = pcd - centroid
        scale = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)))
        # print("scale:",scale)
        pcd = pcd / scale
        pcd_normal = self.points[index][rand_idxs, 3:6]
        # print("scale:",scale)
        max_angle = self.max_angle
        max_trans = self.max_trans / scale
        # print("maxtrans:", max_trans)
        transform = random_pose(max_angle, max_trans / 2)
        pose1 = random_pose(np.pi, max_trans)
        pose2 = transform @ pose1

        if self.partial_rate != 1.0:
            # pcd1, pcd1_normal = Randomcrop(self.partial_rate, pcd, pcd_normal)
            n_points = int(self.partial_rate * self.n_points)
            pcd1, pcd1_normal, gt_mask = farthest_subsample_points(n_points, pcd, pcd_normal)
        else:
            pcd1, pcd1_normal = pcd, pcd_normal

        pcd1 = pcd1 @ pose1[:3, :3].T + pose1[:3, 3]
        pcd1_normal = pcd1_normal @ pose1[:3, :3].T

        pcd2 = pcd @ pose2[:3, :3].T + pose2[:3, 3]
        pcd2_normal = pcd_normal @ pose2[:3, :3].T

        if self.noisy:
            pcd1 = jitter_pcd(pcd1)
            pcd2 = jitter_pcd(pcd2)

        sample = {'points': pcd, 'points_normal': pcd_normal}
        sample['pcd2'] = pcd2
        sample['pcd2_normal'] = pcd2_normal
        sample['transform_gt'] = transform
        sample['scale'] = scale
        sample['pcd1'] = pcd1
        sample['pcd1_normal'] = pcd1_normal

        if self.partial_rate != 1.0:
            sample['gt_mask'] = gt_mask

        #     # inds = np.random.choice(pcd1.shape[0], self.n_points, replace=True)
        #     # sample['pcd1'] = pcd1[inds, :3]
        #     # sample['pcd1_normal'] = pcd1_normal[inds, :3]
        #     # #
        #     res_num = self.n_points - pcd1.shape[0]
        #     residual = np.zeros((res_num, 3))
        #     sample['pcd1'] = np.concatenate([pcd1, residual], axis=0)
        #     sample['pcd1_normal'] = np.concatenate([pcd1_normal, residual], axis=0)
        #     # print(pcd2.shape)
        #     # sample['mask_gt'] = gt_mask
        # else:
        #     sample['pcd1'] = pcd1
        #     sample['pcd1_normal'] = pcd1_normal
        # return pcd1.astype('float32'), pcd2.astype('float32'), pcd1_normal.astype('float32'), pcd2_normal.astype('float32'), transform.astype('float32'), \
        #     centroid.astype('float32'), scale.astype('float32')
        # return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32'), scale.astype('float32')
        return sample

    def __len__(self):
        return len(self.points)

class TestData(Dataset):
    def __init__(self, path, args):
        super(TestData, self).__init__()
        self.pcd1 = get_pcd_data(path='/media/zzy/Data/PCR_Code/DeepBHMR_Med/results_pc/Full/Hip')  # B x M x 6
        self.pcd2 = get_pcd_data(path='/media/zzy/Data/PCR_Code/DeepBHMR_Med/results_pc/Full/Hip')  # B x M x 6
        self.n_points = args.n_points
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        self.noisy = not args.clean
        self.partial_rate = 1.0

    def __getitem__(self, index):

        pcd1 = o3d.io.read_point_cloud('/home/user/桌面/code/HMR_INENET/DeepBHMR&GHMM(1)/DeepBHMR&GHMM/results_pc/Full/Femur/pc1_214.pcd')
        pts1 = np.asarray(pcd1.points)
        pts1_normal = np.asarray(pcd1.normals)

        pcd2 = o3d.io.read_point_cloud('/home/user/桌面/code/HMR_INENET/DeepBHMR&GHMM(1)/DeepBHMR&GHMM/results_pc/Full/Femur/pc2_214.pcd')
        pts2 = np.asarray(pcd2.points)
        pts2_normal = np.asarray(pcd2.normals)

        scale = 58.171504974365234
        pts1 = pts1 / scale
        pts2 = pts2 / scale
        
        res_num = pts2.shape[0] - pts2.shape[0]
        residual = np.zeros((res_num, 3))
        transform = np.load('/media/zzy/Data/PCR_Code/DeepBHMR_Med/results_pc/Full/Femur/T_gt_214.npy')
        transform = transform.reshape(4, 4)
        

        sample = {'points': pts2, 'points_normal': pts2_normal}
        sample['pcd1'] = np.concatenate([pts1, residual], axis=0)
        sample['pcd2'] = pts2
        sample['pcd1_normal'] = np.concatenate([pts1_normal, residual], axis=0)
        sample['pcd2_normal'] = pts2_normal
        sample['transform_gt'] = transform
        sample['scale'] = scale

        return sample




    def __len__(self):
        return len(self.pcd1)


class TestData_ov(Dataset):
    def __init__(self, path, args):
        super(TestData_ov, self).__init__()
        self.pcd1 = get_pcd_data(path='/media/zzy/Data/PCR_Code/DeepBHMR_Med/results_pc/Full/Hip')  # B x M x 6
        self.pcd2 = get_pcd_data(path='/media/zzy/Data/PCR_Code/DeepBHMR_Med/results_pc/Full/Hip')  # B x M x 6
        self.n_points = args.n_points
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        self.noisy = not args.clean
        self.partial_rate = 1.0

    def __getitem__(self, index):
        pcd1 = o3d.io.read_point_cloud('/home/user/桌面/code/HMR_INENET/DeepBHMR&GHMM(1)/DeepBHMR&GHMM/results_pc/Partial/femur/1220/pc1_141.pcd')
        pts1 = np.asarray(pcd1.points)
        pts1_normal = np.asarray(pcd1.normals)

        pcd2 = o3d.io.read_point_cloud('/home/user/桌面/code/HMR_INENET/DeepBHMR&GHMM(1)/DeepBHMR&GHMM/results_pc/Partial/femur/1220/pc2_141.pcd')
        pts2 = np.asarray(pcd2.points)
        pts2_normal = np.asarray(pcd2.normals)

        scale = 58.171504974365234
        pts1 = pts1 / scale
        pts2 = pts2 / scale

        res_num = pts2.shape[0] - pts2.shape[0]
        residual = np.zeros((res_num, 3))
        transform = np.load('/home/user/桌面/code/HMR_INENET/DeepBHMR&GHMM(1)/DeepBHMR&GHMM/results_pc/Partial/femur/1220/T_gt_141.npy')
        transform = transform.reshape(4, 4)

        sample = {'points': pts2, 'points_normal': pts2_normal}
        sample['pcd1'] = np.concatenate([pts1, residual], axis=0)
        sample['pcd2'] = pts2
        sample['pcd1_normal'] = np.concatenate([pts1_normal, residual], axis=0)
        sample['pcd2_normal'] = pts2_normal
        sample['transform_gt'] = transform
        sample['scale'] = scale

        return sample

    def __len__(self):
        return len(self.pcd1)



import argparse

if __name__ == "__main__":
       
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--data_file', type=str, default='/home/user/桌面/code/project/Femur_pcd/train')
   # dataset
    parser.add_argument('--max_angle', type=float, default=45)
    parser.add_argument('--max_trans', type=float, default=500)
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--clean', type=bool, default=True)
    parser.add_argument('--partial', type=float, default=0.5)
    # model
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_clusters', type=int, default=16)
    parser.add_argument('--use_rri', type=bool, default=False)
    parser.add_argument('--use_tnet', type=bool, default=True)
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()

    test_data = TrainData(path=args.data_file, args=args)

    for (src, tgt, src_normal, tgt_normal, T_gt, cen, scale) in test_data:
        print(src.shape)
        print(tgt.shape)
        print(T_gt.shape)
    
    