'''
Copyright (c) 2020 NVIDIA
Author: Wentao Yuan
'''

import argparse
import numpy as np
import torch
import os
from tensorboardX import SummaryWriter
from time import time
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import datetime

# from data_normal import TrainData
from data_pcd import TrainData
from model_normal import DeepHMR
from Overlap import data_utils

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp train_normal.py checkpoints' + '/' + args.exp_name + '/' + 'train_normal.py.backup')
    os.system('cp model_normal.py checkpoints' + '/' + args.exp_name + '/' + 'model_normal.py.backup')
    os.system('cp data_pcd.py checkpoints' + '/' + args.exp_name + '/' + 'data_pcd.py.backup')
    

def train_one_epoch(epoch, model, loader, writer, log_freq, plot_freq):
    model.train()

    log_fmt = 'Epoch {:03d} Step {:03d}/{:03d} Train: ' + \
              'batch time {:.4f}, data time {:.4f}, loss {:.4f}, ' + \
              'rotation error {:.4f}, translation error {:.4f}, RMSE {:.4f}'
    batch_time = []
    data_time = []
    losses = []
    r_errs = []
    t_errs = []
    rmses = []
    total_steps = len(loader)

    start = time()
    # for step, (pts1, pts2, pts1_normal, pts2_normal, T_gt, centroid, scale) in enumerate(loader):
    for step, (input) in enumerate(loader):
        pts1, pts2 = input['pcd1'], input['pcd2']
        pts1_normal, pts2_normal = input['pcd1_normal'], input['pcd2_normal']
        T_gt, scale = input['transform_gt'], input['scale']
        # print("srcshape1:",pts1.shape)
        # print("srcshape2:", pts2.shape)
        # print("src:", pts2)
        if torch.cuda.is_available():
            pts1 = pts1.cuda().float()
            pts2 = pts2.cuda().float()        
            pts1_normal = pts1_normal.cuda().float()
            pts2_normal = pts2_normal.cuda().float()
            T_gt = T_gt.cuda().float()
            scale = scale.cuda().float()
        data_time.append(time() - start)

        model.zero_grad()
        optimizer.zero_grad()
        loss, r_err, t_err, rmse = model(pts1, pts2, pts1_normal, pts2_normal, T_gt, scale)
        loss.backward()
        optimizer.step()
        batch_time.append(time() - start)

        losses.append(loss.item())
        r_errs.append(r_err.mean().item())
        t_errs.append(t_err.mean().item())
        rmses.append(rmse.mean().item())

        global_step = epoch * len(loader) + step + 1

        if global_step % log_freq == 0:
            log_str = log_fmt.format(
                epoch+1, step+1, total_steps,
                np.mean(batch_time), np.mean(data_time), np.mean(losses),
                np.mean(r_errs), np.mean(t_errs), np.mean(rmses)
            )
            textio.cprint(log_str)
            writer.add_scalar('train/loss', np.mean(losses), global_step)
            writer.add_scalar('train/rotation_error', np.mean(r_errs), global_step)
            writer.add_scalar('train/translation_error', np.mean(t_errs), global_step)
            writer.add_scalar('train/RMSE', np.mean(rmses), global_step)
            batch_time.clear()
            data_time.clear()
            losses.clear()
            r_errs.clear()
            t_errs.clear()
            rmses.clear()
        # visualize
        '''
        if global_step % plot_freq == 0:
            fig = model.visualize(0)
            writer.add_figure('train', fig, global_step)
        '''
        start = time()


def eval_one_epoch(epoch, model, loader, writer, global_step, plot_freq):
    model.eval()

    log_fmt = 'Epoch {:03d} Valid: batch time {:.4f}, data time {:.4f}, ' + \
              'loss {:.4f}, rotation error {:.4f}, translation error {:.4f}, RMSE {:.4f}'
    batch_time = []
    data_time = []
    losses = []
    r_errs = []
    t_errs = []
    rmses = []

    start = time()
    # for step, (pts1, pts2, pts1_normal, pts2_normal, T_gt, centroid, scale) in enumerate(tqdm(loader, leave=False)):
    for step, (input) in enumerate(tqdm(loader, leave=False)):
        pts1, pts2 = input['pcd1'], input['pcd2']
        pts1_normal, pts2_normal = input['pcd1_normal'], input['pcd2_normal']
        T_gt, scale = input['transform_gt'], input['scale']

        if torch.cuda.is_available():
            pts1 = pts1.cuda().float()
            pts2 = pts2.cuda().float()        
            pts1_normal = pts1_normal.cuda().float()
            pts2_normal = pts2_normal.cuda().float()
            T_gt = T_gt.cuda().float()
            scale = scale.cuda().float()
        data_time.append(time() - start)

        with torch.no_grad():
            loss, r_err, t_err, rmse = model(pts1, pts2, pts1_normal, pts2_normal, T_gt, scale)
            batch_time.append(time() - start)

        losses.append(loss.item())
        r_errs.append(r_err.mean().item())
        t_errs.append(t_err.mean().item())
        rmses.append(rmse.mean().item())

        start = time()

    log_str = log_fmt.format(
        epoch+1, np.mean(batch_time), np.mean(data_time),
        np.mean(losses), np.mean(r_errs), np.mean(t_errs), np.mean(rmses)
    )
    textio.cprint(log_str)
    writer.add_scalar('valid/loss', np.mean(losses), global_step)
    writer.add_scalar('valid/rotation_error', np.mean(r_errs), global_step)
    writer.add_scalar('valid/translation_error', np.mean(t_errs), global_step)
    writer.add_scalar('valid/RMSE', np.mean(rmses), global_step)

    return np.mean(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general
    # parser.add_argument('--data_file', type=str, default='/root/autodl-tmp/Data/modelnet40_ply_hdf5_2048')
    # parser.add_argument('--categoryfile', type=str, default='/root/autodl-tmp/Data/modelnet40_ply_hdf5_2048/modelnet40_half1.txt')
    parser.add_argument('--train_data_file', type=str, default='/home/user/桌面/code/project/Femur_pcd/train')
    parser.add_argument('--valid_data_file', type=str, default='/home/user/桌面/code/project/Femur_pcd/test')
    # dataset
    parser.add_argument('--max_angle', type=float, default=45)
    parser.add_argument('--max_trans', type=float, default=50)
    parser.add_argument('--partial', type=float, default=0.7)
    parser.add_argument('--n_points', type=int, default=1024)
    parser.add_argument('--clean', type=bool, default=True)
    # model
    parser.add_argument('--d_model', type=int, default=1024)
    parser.add_argument('--n_clusters', type=int, default=16)
    parser.add_argument('--use_rri', type=bool, default=False)
    parser.add_argument('--use_tnet', type=bool, default=True)
    parser.add_argument('--k', type=int, default=20)
    # train setting
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.985)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--plot_freq', type=int, default=250)
    parser.add_argument('--save_freq', type=int, default=30)
    # eval setting
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--eval_plot_freq', type=int, default=10)
    args = parser.parse_args()
    
    args.exp_name = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    _init_(args)
    textio = IOStream('checkpoints/' + args.exp_name + '/run.log')
    textio.cprint(str(args))
    
##################################################################################################################################
##################################################################################################################################
    model = DeepHMR(args)
    if torch.cuda.is_available():
        model.cuda()
    # data = TrainData(args.data_file, args)
    # ids = np.random.permutation(len(data))
    # n_val = int(args.val_fraction * len(data))
    # train_data = Subset(data, ids[n_val:])
    # valid_data = Subset(data, ids[:n_val])

    train_data = TrainData(args.train_data_file, args)
    valid_data = TrainData(args.valid_data_file, args)

    train_loader = DataLoader(train_data, args.batch_size, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_data, args.eval_batch_size, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1)
    writer = SummaryWriter('checkpoints/' + args.exp_name)
    
    best_test_loss = np.inf
##################################################################################################################################
##################################################################################################################################
    for epoch in range(args.n_epochs):
        train_one_epoch(epoch, model, train_loader, writer, args.log_freq, args.plot_freq)
        # scheduler.step()
        global_step = (epoch+1) * len(train_loader)
        
        # validation
        test_loss = eval_one_epoch(epoch, model, valid_loader, writer, global_step, args.eval_plot_freq)
        scheduler.step(test_loss)
##################################################################################################################################
##################################################################################################################################
        for param_group in optimizer.param_groups:
            lr = float(param_group['lr'])
            break
        writer.add_scalar('train/learning_rate', lr, global_step)
        # tensorboard - -logdir = "./checkpoints"
        if (epoch+1) % args.save_freq == 0:
            filename = '{}/checkpoint_epoch-{:d}.pth'.format(('checkpoints/' + args.exp_name + '/models'), epoch+1)
            torch.save(model.state_dict(), filename)
            
        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.pth' % args.exp_name)
