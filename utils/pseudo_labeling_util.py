import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .misc import AverageMeter, accuracy
from .utils import enable_dropout


def pseudo_labeling(args, data_loader, model, itr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    # 伪标签的总体索引
    pseudo_idx = []
    # 伪标签的索引
    pseudo_target = []
    # 伪标签最大标准差
    pseudo_maxstd = []

    gt_target = []
    # 索引列表
    idx_list = []
    gt_list = []
    # 目标列表
    target_list = []
    # 负标签掩码
    nl_mask = []
    model.eval()
    # 如果在伪标签选择中使用不确定性
    if not args.no_uncertainty:
        f_pass = 10
        enable_dropout(model)
    # 使用不确定性
    else:
        f_pass = 1
    # 使用进度条
    if not args.no_progress:
        data_loader = tqdm(data_loader)
    # 不进行梯度计算，提高代码执行效率，只探索前向传播，生成伪标签不需要反向传播
    with torch.no_grad():
        for batch_idx, (inputs, targets, indexs, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            # 输入数据
            inputs = inputs.to(args.device)
            # 目标数据
            targets = targets.to(args.device)
            # 标签输出的概率预测
            out_prob = []
            # 负标签的概率预测
            out_prob_nl = []
            # 每个batch迭代10次
            for _ in range(f_pass):
                # 用标签数据训练好的模型来生成伪标签
                outputs = model(inputs)
                out_prob.append(F.softmax(outputs, dim=1)) #for selecting positive pseudo-labels
                out_prob_nl.append(F.softmax(outputs/args.temp_nl, dim=1)) #for selecting negative pseudo-labels
            # 将输出的概率向量堆叠起来成为矩阵
            out_prob = torch.stack(out_prob)
            out_prob_nl = torch.stack(out_prob_nl)
            # 保存所有迭代次数产生的正标签的标准差
            out_std = torch.std(out_prob, dim=0)
            # 保存所有迭代次数产生的负标签的标准差
            out_std_nl = torch.std(out_prob_nl, dim=0)
            # 保存所有迭代次数产生的标签的均值
            out_prob = torch.mean(out_prob, dim=0)
            out_prob_nl = torch.mean(out_prob_nl, dim=0)
            # 求出softmax输出的概率预测中的最大预测概率， max_idx表示所属的类别序号
            max_value, max_idx = torch.max(out_prob, dim=1)
            # 求出该类别的最大标准差，view表示张量维度重构，-1表示系统自动补齐维度数
            # gather提取向量特定位置的值，在此处表示所有迭代过程中的最大标准差
            max_std = out_std.gather(1, max_idx.view(-1,1))
            out_std_nl = out_std_nl.cpu().numpy()
            
            #selecting negative pseudo-labels
            interm_nl_mask = ((out_std_nl < args.kappa_n) * (out_prob_nl.cpu().numpy() < args.tau_n)) *1

            #manually setting the argmax value to zero 一轮结束后将所有的概率预测向量归零
            for enum, item in enumerate(max_idx.cpu().numpy()):
                interm_nl_mask[enum, item] = 0
            nl_mask.extend(interm_nl_mask)

            idx_list.extend(indexs.numpy().tolist())
            gt_list.extend(targets.cpu().numpy().tolist())
            target_list.extend(max_idx.cpu().numpy().tolist())

            #selecting positive pseudo-labels
            if not args.no_uncertainty:
                selected_idx = (max_value>=args.tau_p) * (max_std.squeeze(1) < args.kappa_p)
            else:
                selected_idx = max_value>=args.tau_p

            pseudo_maxstd.extend(max_std.squeeze(1)[selected_idx].cpu().numpy().tolist())
            pseudo_target.extend(max_idx[selected_idx].cpu().numpy().tolist())
            pseudo_idx.extend(indexs[selected_idx].numpy().tolist())
            gt_target.extend(targets[selected_idx].cpu().numpy().tolist())

            loss = F.cross_entropy(outputs, targets.to(dtype=torch.long))
            prec1, prec5 = accuracy(outputs[selected_idx], targets[selected_idx], topk=(1, 5))

            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                data_loader.set_description("Pseudo-Labeling Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(data_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            data_loader.close()

    pseudo_target = np.array(pseudo_target)
    gt_target = np.array(gt_target)
    pseudo_maxstd = np.array(pseudo_maxstd)
    pseudo_idx = np.array(pseudo_idx)

    #class balance the selected pseudo-labels
    # 类平衡选定的伪标签
    if itr < args.class_blnc-1:
        min_count = 5000000 #arbitary large value
        for class_idx in range(args.num_classes):
            class_len = len(np.where(pseudo_target==class_idx)[0])
            if class_len < min_count:
                min_count = class_len
        #  当某一类别的最小计数非常低时，这25用于避免去甲化情况
        min_count = max(25, min_count) #this 25 is used to avoid degenarate cases when the minimum count for a certain class is very low

        blnc_idx_list = []
        for class_idx in range(args.num_classes):
            current_class_idx = np.where(pseudo_target==class_idx)
            if len(np.where(pseudo_target==class_idx)[0]) > 0:
                current_class_maxstd = pseudo_maxstd[current_class_idx]
                sorted_maxstd_idx = np.argsort(current_class_maxstd)
                current_class_idx = current_class_idx[0][sorted_maxstd_idx[:min_count]] #select the samples with lowest uncertainty 
                blnc_idx_list.extend(current_class_idx)

        blnc_idx_list = np.array(blnc_idx_list)
        pseudo_target = pseudo_target[blnc_idx_list]
        pseudo_idx = pseudo_idx[blnc_idx_list]
        gt_target = gt_target[blnc_idx_list]

    pseudo_labeling_acc = (pseudo_target == gt_target)*1
    pseudo_labeling_acc = (sum(pseudo_labeling_acc)/len(pseudo_labeling_acc))*100
    print(f'Pseudo-Labeling Accuracy (positive): {pseudo_labeling_acc}, Total Selected: {len(pseudo_idx)}')

    pseudo_nl_mask = []
    pseudo_nl_idx = []
    nl_gt_list = []

    for i in range(len(idx_list)):
        if idx_list[i] not in pseudo_idx and sum(nl_mask[i]) > 0:
            pseudo_nl_mask.append(nl_mask[i])
            pseudo_nl_idx.append(idx_list[i])
            nl_gt_list.append(gt_list[i])

    nl_gt_list = np.array(nl_gt_list)
    pseudo_nl_mask = np.array(pseudo_nl_mask)
    one_hot_targets = np.eye(args.num_classes)[nl_gt_list]
    one_hot_targets = one_hot_targets - 1
    one_hot_targets = np.abs(one_hot_targets)
    flat_pseudo_nl_mask = pseudo_nl_mask.reshape(1,-1)[0]
    flat_one_hot_targets = one_hot_targets.reshape(1,-1)[0]
    flat_one_hot_targets = flat_one_hot_targets[np.where(flat_pseudo_nl_mask == 1)]
    flat_pseudo_nl_mask = flat_pseudo_nl_mask[np.where(flat_pseudo_nl_mask == 1)]

    nl_accuracy = (flat_pseudo_nl_mask == flat_one_hot_targets)*1
    nl_accuracy_final = (sum(nl_accuracy)/len(nl_accuracy))*100
    print(f'Pseudo-Labeling Accuracy (negative): {nl_accuracy_final}, Total Selected: {len(nl_accuracy)}, Unique Samples: {len(pseudo_nl_mask)}')
    pseudo_label_dict = {'pseudo_idx': pseudo_idx.tolist(), 'pseudo_target':pseudo_target.tolist(), 'nl_idx': pseudo_nl_idx, 'nl_mask': pseudo_nl_mask.tolist()}
 
    return losses.avg, top1.avg, pseudo_labeling_acc, len(pseudo_idx), nl_accuracy_final, len(nl_accuracy), len(pseudo_nl_mask), pseudo_label_dict