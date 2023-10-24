import argparse
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
from collections import OrderedDict
import pickle
import numpy as np
from re import search
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from data.cifar import get_cifar10, get_cifar100
from data.officehome import get_office_home
from utils import AverageMeter, accuracy
from utils.utils import *
from utils.train_util import train_initial, train_regular
from utils.evaluate import test
from utils.pseudo_labeling_util import pseudo_labeling


def main():
    run_started = datetime.today().strftime('%d-%m-%y_%H%M') #start time to create unique experiment name
    parser = argparse.ArgumentParser(description='UPS Training')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--dataset', default='office_home', type=str,
                        choices=['cifar10', 'cifar100', 'office_home'],
                        help='dataset names')
    # parser.add_argument('--n-lbl', type=int, default=4000,
    #                     help='number of labeled data')
    parser.add_argument('--n-lbl', type=int, default=930,
                        help='number of labeled data')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'cnn13', 'shakeshake'],
                        help='architecture name')
    # 代表要进行多少次batch_size的迭代，十框石头一次挑一框，batch_idx即为10
    parser.add_argument('--iterations', default=20, type=int,
                        help='number of total pseudo-labeling iterations to run')
    # 把所有的训练数据全部迭代遍历一遍（单次epoch）
    # parser.add_argument('--epchs', default=1024, type=int,
    #                     help='number of total epochs to run')
    parser.add_argument('--epchs', default=24, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    # 代表每次从所有数据中取出一小筐子数据进行训练，类似挑十框石头，每次挑一筐，此时的batch_size=1
    parser.add_argument('--batchsize', default=128, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate, default 0.03')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='dropout probs')
    parser.add_argument('--num-classes', default=10, type=int,
                        help='total classes')
    parser.add_argument('--class-blnc', default=10, type=int,
                        help='total number of class balanced iterations')
    parser.add_argument('--tau-p', default=0.70, type=float,
                        help='confidece threshold for positive pseudo-labels, default 0.70')
    parser.add_argument('--tau-n', default=0.05, type=float,
                        help='confidece threshold for negative pseudo-labels, default 0.05')
    parser.add_argument('--kappa-p', default=0.05, type=float,
                        help='uncertainty threshold for positive pseudo-labels, default 0.05')
    parser.add_argument('--kappa-n', default=0.005, type=float,
                        help='uncertainty threshold for negative pseudo-labels, default 0.005')
    parser.add_argument('--temp-nl', default=2.0, type=float,
                        help='temperature for generating negative pseduo-labels, default 2.0')
    parser.add_argument('--no-uncertainty', action='store_true',
                        help='use uncertainty in the pesudo-label selection, default true')
    # 用于区分不同实验的额外文本。它还创建一个新的标记/未标记拆分
    parser.add_argument('--split-txt', default='run1', type=str,
                        help='extra text to differentiate different experiments. it also creates a new labeled/unlabeled split')
    parser.add_argument('--model-width', default=2, type=int,
                        help='model width for WRN-28')
    parser.add_argument('--model-depth', default=28, type=int,
                        help='model depth for WRN')
    parser.add_argument('--test-freq', default=10, type=int,
                        help='frequency of evaluations')
    
    args = parser.parse_args()
    #print key configurations
    print('########################################################################')
    print('########################################################################')
    print(f'dataset:                                  {args.dataset}')
    print(f'number of labeled samples:                {args.n_lbl}')
    print(f'architecture:                             {args.arch}')
    print(f'number of pseudo-labeling iterations:     {args.iterations}')
    print(f'number of epochs:                         {args.epchs}')
    print(f'batch size:                               {args.batchsize}')
    print(f'lr:                                       {args.lr}')
    print(f'value of tau_p:                           {args.tau_p}')
    print(f'value of tau_n:                           {args.tau_n}')
    print(f'value of kappa_p:                         {args.kappa_p}')
    print(f'value of kappa_n:                         {args.kappa_n}')
    print('########################################################################')
    print('########################################################################')

    DATASET_GETTERS = {'cifar10': get_cifar10, 'cifar100': get_cifar100, 'office_home': get_office_home}
    exp_name = f'exp_{args.dataset}_{args.n_lbl}_{args.arch}_{args.split_txt}_{args.epchs}_{args.class_blnc}_{args.tau_p}_{args.tau_n}_{args.kappa_p}_{args.kappa_n}_{run_started}'
    device = torch.device('cuda', args.gpu_id)
    args.device = device
    args.exp_name = exp_name
    args.dtype = torch.float32
    if args.seed != -1:
        set_seed(args)
    args.out = os.path.join(args.out, args.exp_name)
    start_itr = 0

    if args.resume and os.path.isdir(args.resume):
        resume_files = os.listdir(args.resume)
        resume_itrs = [int(item.replace('.pkl','').split("_")[-1]) for item in resume_files if 'pseudo_labeling_iteration' in item]
        if len(resume_itrs) > 0:
            start_itr = max(resume_itrs)
        args.out = args.resume
    os.makedirs(args.out, exist_ok=True)
    writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'office_home':
        args.num_classes = 65
    
    for itr in range(start_itr, args.iterations):
        if itr == 0 and args.n_lbl < 4000: #use a smaller batchsize to increase the number of iterations
            args.batch_size = 64
            args.epochs = 1024
        else:
            args.batch_size = args.batchsize
            args.epochs = args.epchs

        if os.path.exists(f'data/splits/{args.dataset}_basesplit_{args.n_lbl}_{args.split_txt}.pkl'):
            lbl_unlbl_split = f'data/splits/{args.dataset}_basesplit_{args.n_lbl}_{args.split_txt}.pkl'
        else:
            lbl_unlbl_split = None
        
        #load the saved pseudo-labels
        # 当迭代次数大于0时才存在模型设置的伪标签
        if itr > 0:
            # 伪标签数据的保存路径
            pseudo_lbl_dict = f'{args.out}/pseudo_labeling_iteration_{str(itr)}.pkl'
        else:
            pseudo_lbl_dict = None
        # 获取标签数据、负标签数据、无标签数据、测试数据 --> 将伪标签数据中的多少变为正标签数据，和负标签数据
        lbl_dataset, nl_dataset, unlbl_dataset, test_dataset = DATASET_GETTERS[args.dataset]('data/datasets', args.n_lbl,
                                                                lbl_unlbl_split, pseudo_lbl_dict, itr, args.split_txt)

        model = create_model(args)
        model.to(args.device)
        # 负标签数据的batchsize计算，使用负标签进行负标签训练
        nl_batchsize = int((float(args.batch_size) * len(nl_dataset))/(len(lbl_dataset) + len(nl_dataset)))

        # 第一次迭代的时候还没有负标签数据
        if itr == 0:
            lbl_batchsize = args.batch_size
            # 迭代次数等于总数据量除以batch_size
            args.iteration = len(lbl_dataset) // args.batch_size
        else:
            # 标签数据的batchsize等于模型batch_size - 负标签数据的batchsize
            lbl_batchsize = args.batch_size - nl_batchsize
            # 迭代次数等于标签数据和负标签数据量之和除以模型的batchsize,确保所有的数据都会被使用
            args.iteration = (len(lbl_dataset) + len(nl_dataset)) // args.batch_size
        # 加载正标签数据
        lbl_loader = DataLoader(
            lbl_dataset,
            sampler=RandomSampler(lbl_dataset),
            batch_size=lbl_batchsize,
            num_workers=args.num_workers,
            drop_last=True)
        # 加载负标签数据
        nl_loader = DataLoader(
            nl_dataset,
            sampler=RandomSampler(nl_dataset),
            batch_size=nl_batchsize,
            num_workers=args.num_workers,
            drop_last=True)
        # 加载测试数据
        test_loader = DataLoader(
            test_dataset,
            sampler=SequentialSampler(test_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)
        # 加载无标签数据
        unlbl_loader = DataLoader(
            unlbl_dataset,
            sampler=SequentialSampler(unlbl_dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers)
        # 优化器，使用随机梯度下降算法
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=args.nesterov)
        # 训练的总体步骤等于 epoche与迭代次数乘积，等于模型需要更新训练参数的次数，每一次批训练都需要更新参数
        args.total_steps = args.epochs * args.iteration
        scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup * args.iteration, args.total_steps)
        start_epoch = 0
        # resume -> 最新检查点的路径
        if args.resume and itr == start_itr and os.path.isdir(args.resume):
            resume_itrs = [int(item.replace('.pth.tar','').split("_")[-1]) for item in resume_files if 'checkpoint_iteration_' in item]
            if len(resume_itrs) > 0:
                checkpoint_itr = max(resume_itrs)
                resume_model = os.path.join(args.resume, f'checkpoint_iteration_{checkpoint_itr}.pth.tar')
                if os.path.isfile(resume_model) and checkpoint_itr == itr:
                    checkpoint = torch.load(resume_model)
                    best_acc = checkpoint['best_acc']
                    start_epoch = checkpoint['epoch']
                    model.load_state_dict(checkpoint['state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])

        model.zero_grad()
        best_acc = 0
        for epoch in range(start_epoch, args.epochs):
            # 第一次迭代还没有负标签，所以不需要加载nl_loader
            if itr == 0:
                # 训练损失，有标签数据+无标签数据的训练过程
                train_loss = train_initial(args, lbl_loader, model, optimizer, scheduler, epoch, itr)
            else:
                train_loss = train_regular(args, lbl_loader, nl_loader, model, optimizer, scheduler, epoch, itr)

            test_loss = 0.0
            test_acc = 0.0
            test_model = model
            if epoch > (args.epochs+1)/2 and epoch%args.test_freq==0:
                test_loss, test_acc = test(args, test_loader, test_model)
            elif epoch == (args.epochs-1):
                test_loss, test_acc = test(args, test_loader, test_model)
            # 绘制图像
            writer.add_scalar('train/1.train_loss', train_loss, (itr*args.epochs)+epoch)
            writer.add_scalar('test/1.test_acc', test_acc, (itr*args.epochs)+epoch)
            writer.add_scalar('test/2.test_loss', test_loss, (itr*args.epochs)+epoch)

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out, f'iteration_{str(itr)}')
    
        checkpoint = torch.load(f'{args.out}/checkpoint_iteration_{str(itr)}.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        model.zero_grad()

        #pseudo-label generation and selection 生成伪标签，并且选择符合条件的伪标签进入下一次迭代训练，最关键的地方，确定伪标签如何处理的函数,伪标签损失
        pl_loss, pl_acc, pl_acc_pos, total_sel_pos, pl_acc_neg, total_sel_neg, unique_sel_neg, pseudo_label_dict = pseudo_labeling(args, unlbl_loader, model, itr)

        writer.add_scalar('pseudo_labeling/1.regular_loss', pl_loss, itr)
        writer.add_scalar('pseudo_labeling/2.regular_acc', pl_acc, itr)
        writer.add_scalar('pseudo_labeling/3.pseudo_acc_positive', pl_acc_pos, itr)
        writer.add_scalar('pseudo_labeling/4.total_sel_positive', total_sel_pos, itr)
        writer.add_scalar('pseudo_labeling/5.pseudo_acc_negative', pl_acc_neg, itr)
        writer.add_scalar('pseudo_labeling/6.total_sel_negative', total_sel_neg, itr)
        writer.add_scalar('pseudo_labeling/7.unique_samples_negative', unique_sel_neg, itr)

        # 将伪标签数据保存到.pkl文件
        with open(os.path.join(args.out, f'pseudo_labeling_iteration_{str(itr+1)}.pkl'),"wb") as f:
            pickle.dump(pseudo_label_dict,f)
        # 保存模型训练日志信息
        with open(os.path.join(args.out, 'log.txt'), 'a+') as ofile:
            ofile.write(f'############################# PL Iteration: {itr+1} #############################\n')
            ofile.write(f'Last Test Acc: {test_acc}, Best Test Acc: {best_acc}\n')
            ofile.write(f'PL Acc (Positive): {pl_acc_pos}, Total Selected (Positive): {total_sel_pos}\n')
            ofile.write(f'PL Acc (Negative): {pl_acc_neg}, Total Selected (Negative): {total_sel_neg}, Unique Negative Samples: {unique_sel_neg}\n\n')

    writer.close()


if __name__ == '__main__':
    cudnn.benchmark = True
    main()