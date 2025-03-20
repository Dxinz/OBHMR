'''
Copyright (c) 2024 SUD
Author: Xinzhe Du
'''

import argparse
import numpy as np
import os
import torch
from time import time
from torch.utils.data import DataLoader
from tqdm import tqdm

# from data_normal import TrainData
from data_pcd import TrainData, TestData
from bi_model_normal_Transformer import DeepHMR
import open3d as o3d
import trimesh


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def evaluate(model, loader, rmse_thresh, save_results=False, results_dir=None):
    model.eval()

    log_fmt = 'Test: inference time {:.3f}, preprocessing time {:.3f}, loss {:.4f}, ' + \
              'rotation error {:.4f}, translation error {:.4f}, RMSE {:.4f}, Recall {:.4f}'
    inference_time = 0
    preprocess_time = 0
    losses = 0
    r_errs = 0
    t_errs = 0
    rmses = 0
    n_correct = 0
    N = 0
    index = 0
    m = 0

    # if save_results:
    #     rotations = []
    #     translations = []
    #     rotations_gt = []
    #     translations_gt = []
    r_err_list = []
    t_err_list = []
    rmse_list = []
    start = time()
    for step, (input) in enumerate(tqdm(loader, leave=False)):
    # for step, (pts1, pts2, pts1_normal, pts2_normal, T_gt, centroid, scale) in enumerate(tqdm(loader, leave=False)):
        pts1, pts2 = input['pcd1'][:, :717, :], input['pcd2']
        pts1_normal, pts2_normal = input['pcd1_normal'][:, :717, :], input['pcd2_normal']
        T_gt, scale = input['transform_gt'], input['scale']
        if torch.cuda.is_available():
            pts1 = pts1.cuda().float()
            pts2 = pts2.cuda().float()
            T_gt = T_gt.cuda().float()
            pts1_normal = pts1_normal.cuda().float()
            pts2_normal = pts2_normal.cuda().float()
            scale = scale.cuda().float()
        preprocess_time += time() - start
        N += pts1.shape[0]
        # m += 1
        # print(m)

        start = time()
        with torch.no_grad():
            loss, r_err, t_err, rmse = model(pts1, pts2, pts1_normal, pts2_normal, T_gt, scale)
            # loss, r_err, t_err, rmse, T_pred, initial_r, initial_t = model(pts1, pts2, pts1_normal, pts2_normal, T_gt)
            inference_time += time() - start
            # list = [146, 203, 217, 260, 295, 340, 432, 495, 530, 539]
            
            # if r_err.item() < 1 and t_err.item() < 1 and r_err.item() != 0:
            # # if step == 145:
            #     print(step, ', RMSE: ', rmse.item(), '\t Scale: ', scale.item())
            #     print('Initial Rot: ', model.initial_r.item(), '\t', 'Initial trans: ', model.initial_t.item())
            #     print('Rot err: ', r_err.item(), '\t', 'trans err: ', t_err.item())
            #     pts1_mm = pts1 * scale.unsqueeze(-1).unsqueeze(-1)
            #     pts2_mm = pts2 * scale.unsqueeze(-1).unsqueeze(-1)
            #     pts1_np, pts1_normal_np = pts1_mm.squeeze(0).cpu().numpy(), pts1_normal.squeeze(0).cpu().numpy()
            #     pts2_np, pts2_normal_np = pts2_mm.squeeze(0).cpu().numpy(), pts2_normal.squeeze(0).cpu().numpy()
            #     T_pred_np = model.T_12_mm.squeeze(0).cpu().numpy()

            #     pts1_pc = trimesh.points.PointCloud(pts1_np)
            #     pts1_pc.colors = [255, 77, 0]

            #     pts2_pc = o3d.geometry.PointCloud()
            #     pts2_pc.points = o3d.utility.Vector3dVector(pts2_np)
            #     pts2_pc.normals = o3d.utility.Vector3dVector(pts2_normal_np)
            #     pts2_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pts2_pc, depth=9)
            #     pts2_mesh = trimesh.Trimesh(np.asarray(pts2_mesh.vertices), np.asarray(pts2_mesh.triangles), vertex_normals=np.asarray(pts2_mesh.vertex_normals))
            #     pts2_mesh.visual.face_colors = [65, 105, 255, 200]
            #     scence = trimesh.Scene([pts1_pc, pts2_mesh])
            #     scence.show(line_settings={'point_size':18})

            #     pts1_reg_np = pts1_np @ T_pred_np[:3, :3].T + T_pred_np[:3, 3]
            #     pts1_reg_pc = trimesh.points.PointCloud(pts1_reg_np)
            #     pts1_reg_pc.colors = [255, 77, 0]
            #     pts1_normal_reg_np = pts1_normal.squeeze(0).cpu().numpy() @ T_pred_np[:3, :3].T
            #     scence = trimesh.Scene([pts1_reg_pc, pts2_mesh])
            #     scence.show(line_settings={'point_size':18})
                
            #     pts1_pc = o3d.geometry.PointCloud()
            #     pts1_pc.points = o3d.utility.Vector3dVector(pts1_np)
            #     pts1_pc.normals = o3d.utility.Vector3dVector(pts1_normal_np)
            #     pc1_path = '/media/zzy/Data/PCR_Code/DeepBHMR_Med/results_pc/Full/Femur/pc1_' + str(step) + '.pcd'
            #     o3d.io.write_point_cloud(pc1_path, pts1_pc)
            #     pts2_pc = o3d.geometry.PointCloud()
            #     pts2_pc.points = o3d.utility.Vector3dVector(pts2_np)
            #     pts2_pc.normals = o3d.utility.Vector3dVector(pts2_normal_np)
            #     pc2_path = '/media/zzy/Data/PCR_Code/DeepBHMR_Med/results_pc/Full/Femur/pc2_' + str(step) + '.pcd'
            #     o3d.io.write_point_cloud(pc2_path, pts2_pc)
            #     pts1_reg_pc = o3d.geometry.PointCloud()
            #     pts1_reg_pc.points = o3d.utility.Vector3dVector(pts1_reg_np)
            #     pts1_reg_pc.normals = o3d.utility.Vector3dVector(pts1_normal_reg_np)
            #     pc1_reg_path = '/media/zzy/Data/PCR_Code/DeepBHMR_Med/results_pc/Full/Femur/pc1_reg_' + str(step) + '.pcd'
            #     o3d.io.write_point_cloud(pc1_reg_path, pts1_reg_pc)
            #     T_gt_path = '/media/zzy/Data/PCR_Code/DeepBHMR_Med/results_pc/Full/Femur/T_gt_' + str(step) + '.npy'
            #     np.save(os.path.join(T_gt_path), np.concatenate(T_gt.squeeze(0).cpu().numpy(), 0))
                
        losses += loss.item()
        r_errs += r_err.sum().item()
        t_errs += t_err.sum().item()
        rmses += rmse.sum().item()
        n_correct += (rmse < rmse_thresh).sum().item()
        r_err_list.append(r_err.cpu().numpy())
        t_err_list.append(t_err.cpu().numpy())
        rmse_list.append(rmse.cpu().numpy())

        # if save_results:
        #     rotations.append(model.T_12[:, :3, :3].cpu().numpy())
        #     translations.append(model.T_12[:, :3, 3].cpu().numpy())
        #     rotations_gt.append(T_gt[:, :3, :3].cpu().numpy())
        #     translations_gt.append(T_gt[:, :3, 3].cpu().numpy())

        start = time()

    log_str = log_fmt.format(
        inference_time / N, preprocess_time / N, losses / len(loader),
        r_errs / N, t_errs / N, rmses / N, n_correct / N
    )
    print(log_str)
    print(format(np.mean(r_err_list), "0.4f"), '±', format(np.std(r_err_list), "0.4f"))
    print(format(np.mean(t_err_list), "0.4f"), '±', format(np.std(t_err_list), "0.4f"))
    print(format(np.mean(rmse_list), "0.4f"), '±', format(np.std(rmse_list), "0.4f"))
    np.save(os.path.join(results_dir, 'Parial_Femur_RMSE_Translation.npy'), np.concatenate(rmse_list, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--data_file', type=str, default='/home/user/桌面/code/project/Femur_pcd/test')
    # parser.add_argument('--data_file', type=str, default='/root/autodl-tmp/Data/modelnet40_ply_hdf5_2048')
    # parser.add_argument('--categoryfile', type=str, default='/root/autodl-tmp/Data/modelnet40_ply_hdf5_2048/modelnet40_half2.txt')
    parser.add_argument('--results_dir', type=str,
                        default='/home/user/桌面/code/project/Femur_pcd/results/2024-01-30_13-08')
    parser.add_argument('--checkpoint', type=str,
                        default='/home/user/桌面/code/HMR_INENET/DeepBHMR&GHMM(1)/DeepBHMR&GHMM/checkpoints/2024-02-04_15-54/models/best_model.pth')
    parser.add_argument('--save_results', type=str, default=False)
    parser.add_argument('--rmse_thresh', type=int, default=1)
    # dataset
    parser.add_argument('--max_angle', type=float, default=45)
    parser.add_argument('--max_trans', type=float, default=50)
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--clean', type=bool, default=False)
    parser.add_argument('--partial', type=float, default=0.7)
    # model
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_clusters', type=int, default=16)
    parser.add_argument('--use_rri', type=bool, default=False)
    parser.add_argument('--use_tnet', type=bool, default=True)  
    parser.add_argument('--k', type=int, default=20)
    args = parser.parse_args()
    print(args)
    
    model = DeepHMR(args)
    if torch.cuda.is_available():
        model.cuda()

    test_data = TrainData(args.data_file, args)
    test_loader = DataLoader(test_data, args.batch_size)

    model.load_state_dict(torch.load(args.checkpoint))
    evaluate(model, test_loader, args.rmse_thresh, args.save_results, args.results_dir)
