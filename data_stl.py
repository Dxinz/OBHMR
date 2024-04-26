'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import h5py
import numpy as np
import os
import torch
from pycuda import gpuarray
from pycuda.compiler import SourceModule
from scipy.spatial import cKDTree
from torch.utils.data import Dataset
import open3d as o3d
import glob
import math


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
    A = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    # R = np.eye(3) + np.sin(math.pi * 2 / 3) * A + (1 - np.cos(math.pi * 2 / 3)) * np.dot(A, A)
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    # t *= 0.6
    return np.expand_dims(t, 1)

def _read_h5_files(fnames, categories):
    '''
       copied from RPM codes

       all_data:   B x M x 6 or B x N x 6
       all_labels: (B,)
    '''

    all_data = []
    all_labels = []

    for fname in fnames:
        f = h5py.File(fname, mode='r')
        pts        =  f['data'][:]
        pts_normal =  f['normal'][:]
        
        data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
        labels = f['label'][:].flatten().astype(np.int64)
        
        if categories is not None:  # Filter out unwanted categories
            mask = np.isin(labels, categories).flatten()
            data = data[mask, ...]
            labels = labels[mask, ...]
      
        all_data.append(data)
        all_labels.append(labels)

    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_data, all_labels 


def get_data_normal(dataset_path, subset, categoryfile):
   '''
    inputs: 
            dataset_path:        the path for the train/test data, e.g., '/home/zmin/rpmnet/RPMNet/Data/modelnet40_ply_hdf5_2048/'
            subset:              'train' or 'test'
            categoryfile:        the path for the used categories
     outputs:
             _data: B x M x 6. For example, 5112 x 2048 x 6 for the first half, 4728 x 2048 x 6 for the second half. 
   '''
   with open(os.path.join(dataset_path, '{}_files.txt'.format(subset))) as fid:
       h5_filelist = [line.strip() for line in fid]
       h5_filelist = [x.replace('data/modelnet40_ply_hdf5_2048/', '') for x in h5_filelist]
       h5_filelist = [os.path.join(dataset_path, f) for f in h5_filelist]

   with open(os.path.join(dataset_path, 'shape_names.txt')) as fid:
       _classes = [l.strip() for l in fid]
       _category2idx = {e[1]: e[0] for e in enumerate(_classes)}

   # get the corresponding categories   
   categories = [line.rstrip('\n') for line in open(categoryfile)]
   categories.sort()
   # convert the categories into ids
   categories_idx = [_category2idx[c] for c in categories]
   # get the data
   _data, _labels = _read_h5_files(h5_filelist, categories_idx)   

   return _data

def get_femur_data(path):
    
    all_data = []
    
    for stlfile_name in glob.glob(os.path.join(path, '*.stl')):
        mesh = o3d.io.read_triangle_mesh(stlfile_name)
        V_mesh = np.asarray(mesh.vertices)
        V_normal_mesh = np.asarray(mesh.vertex_normals)
        all_data.append(np.concatenate([V_mesh, V_normal_mesh], axis=-1))
    
    return all_data


class TrainData(Dataset):
    def __init__(self, path, args):
        super(TrainData, self).__init__()
        self.points = get_femur_data(path)  # B x M x 6
        self.n_points = args.n_points
        # self.sample = args.sample
        self.max_angle = args.max_angle / 180 * np.pi
        self.max_trans = args.max_trans
        self.noisy = not args.clean
        self.use_rri = args.use_rri

    def __getitem__(self, index):
        # rand_idxs = np.random.choice(self.points.shape[1], self.n_points, replace=False)
        if self.points[index].shape[0] < self.n_points:
            rand_idxs = np.random.choice(self.points[index].shape[0], self.n_points, replace=True)
        else:
            rand_idxs = np.random.choice(self.points[index].shape[0], self.n_points, replace=False)

        # if self.sample is not 1.0:
        #     rand_idxs_sample = np.random.choice(self.n_points, int(self.sample*self.n_points), replace=False)
        # else:
        #     rand_idxs_sample = rand_idxs

        pcd = self.points[index][rand_idxs,:3]
        centroid = np.mean(pcd, axis=0)
        pcd = pcd - centroid
        
        scale = np.max(np.sqrt(np.sum(pcd ** 2, axis=1)))
        pcd1 = (pcd) / scale
        
        ''' needs to be fulfilled'''
        pcd_normal1 = self.points[index][rand_idxs,3:6]
        
        transform = random_pose(self.max_angle, self.max_trans)
        pcd2 = pcd @ transform[:3, :3].T + transform[:3, 3]
        pcd2 = (pcd2) / scale
        
        pcd_normal2 = pcd_normal1 @ transform[:3, :3].T
        
        if self.noisy:
            pcd1 = jitter_pcd(pcd1)
            pcd2 = jitter_pcd(pcd2)
            
        
        return pcd1.astype('float32'), pcd2.astype('float32'), pcd_normal1.astype('float32'), pcd_normal2.astype('float32'), transform.astype('float32'), \
            centroid.astype('float32'), scale.astype('float32')
        # return pcd1.astype('float32'), pcd2.astype('float32'), transform.astype('float32'), scale.astype('float32')

    def __len__(self):
        return len(self.points)


import argparse

if __name__ == "__main__":
    
    # mesh = o3d.io.read_triangle_mesh('/media/zzy/PolyU_ZZY/1_datasets/MedShapeNet/femur/s0238_femur_right.nii.g_1.stl')
    # V_mesh = np.asarray(mesh.vertices)
    # V_normal_mesh = np.asarray(mesh.vertex_normals)
    # V_data = np.concatenate([V_mesh, V_normal_mesh], axis=-1)
    
    
    # print(V_mesh.shape)
    
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--data_file', type=str, default='/media/zzy/PolyU_ZZY/1_datasets/MedShapeNet/femur')
    # parser.add_argument('--data_file', type=str, default='/home/zzy/PCR/DATA/modelnet40_ply_hdf5_2048')
    # parser.add_argument('--categoryfile', type=str, default='/home/zzy/PCR/RPMNet/src/data_loader/modelnet40_half1.txt')
    # dataset
    parser.add_argument('--max_angle', type=float, default=180)
    parser.add_argument('--max_trans', type=float, default=1.0)
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--clean', type=bool, default=True)
    # model
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_clusters', type=int, default=16)
    parser.add_argument('--use_rri', type=bool, default=False)
    parser.add_argument('--use_tnet', type=bool, default=True)  
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()
    
    test_data = TrainData(path=args.data_file, args=args)
    
    for (src, tgt, T_gt) in test_data:
        print(src.shape)
        print(tgt.shape)
        print(T_gt.shape)
    
    