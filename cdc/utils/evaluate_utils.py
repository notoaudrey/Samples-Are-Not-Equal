"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import wandb 
import math
import random
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, roc_auc_score,  precision_recall_curve, auc, roc_curve, silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from openTSNE import TSNE as OpenTSNE
import umap
import faiss
from typing import Dict, List

from torchmetrics.functional import calibration_error
from scipy.stats import gaussian_kde
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick
from sklearn.manifold import TSNE
from collections import Counter, defaultdict
from scipy.optimize import linear_sum_assignment
import seaborn as sns
import pandas as pd
import cv2
import torchvision.transforms as transforms
import cdc.utils.misc as misc
import pdb
from scipy.stats import norm
# from validclust import dunn
# import hdbscan



def get_feature_dimensions_backbone(cfg):
    if cfg['backbone']['name'] == 'resnet18':
        return 512
    elif cfg['backbone']['name'] == 'resnet34':
        return 512
    elif cfg['backbone']['name'] == 'resnet50':
        return 2048
    else:
        raise NotImplementedError

@torch.no_grad()
def get_predictions(cfg, dataloader, model, return_features=False, is_train=False, cali_mlp = None, return_input=False, longtail=False, ratio=1, selector=None, propos=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    if cali_mlp is not None:
        cali_mlp.eval()
    predictions = [[] for _ in range(cfg['backbone']['nheads'])]
    probs = [[] for _ in range(cfg['backbone']['nheads'])]
    output_val = [[] for _ in range(cfg['backbone']['nheads'])]
    if 'divclust' in cfg['name']:
        predictions = [[] for _ in range(cfg['clusterings'])]
        probs = [[] for _ in range(cfg['clusterings'])]
    targets = []
    if return_features:
        if 'spicewide3' in cfg['name']:
            if 'cifar10' in cfg['name']:
                ft_dim = 128
            if 'cifar20' in cfg['name']:
                ft_dim = 512
            if 'stl10' in cfg['name']:
                ft_dim = 256

        elif propos:
            ft_dim=256
        else:
            ft_dim = get_feature_dimensions_backbone(cfg)
            
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()
    if return_input:
        s1,s2,s3,s4 = dataloader.dataset.data.shape
        inputs = torch.zeros((s1, s2*s3*s4)).cuda()
    """ if is_train:
        output_val = [] """
    
    #import pdb; pdb.set_trace()
    
    ptr = 0
    # print("checkpoint 1.1")
    with torch.no_grad():
        for batch in dataloader:
            if is_train:
                images = batch['image'].cuda(non_blocking=True)
                targets_ = batch['target'].cuda(non_blocking=True)
            else:
                images, targets_ = batch[0], batch[1]
                if len(batch) > 2:
                    indices = batch[2]
            bs = images.shape[0]
            
            #import pdb; pdb.set_trace()

            if cali_mlp is not None:
                fea = model(images.cuda(non_blocking=True),
                                     forward_pass='backbone')
                output = [cali_mlp(fea, forward_pass='calibration')]
                s_output = output

                if propos:
                    fea = model(images.cuda(non_blocking=True),
                                     forward_pass='backbone_propos')
                    output = [cali_mlp(fea, forward_pass='calibration_propos')]
                    s_output = output
                
                """ if longtail:
                    tail_output = cali_mlp.module.calibration_tail(fea)
                    medium_output = cali_mlp.module.calibration_tail(fea)
                    output = [
                        head_output + tail_output + medium_output
                        for head_output in output
                    ]
                    s_output = output """
                
            elif 'divclust' in cfg['name']:
                res = model(images.cuda(non_blocking=True),
                        forward_pass='return_all')
                output = res['output']
                
            else:
                res = model(images.cuda(non_blocking=True),
                        forward_pass='return_all')
                output = res['output']
                s_output = res['output']

                if propos:
                    res = model(images.cuda(non_blocking=True),
                        forward_pass='propos_all')
                    output = res['output']
                    s_output = res['output']
                
                if selector is not None:
                    selector.update(indices, output[0])
                
                if longtail:
                    #print('longtail')
                    tail_output = model.module.classify_tail(res['features'])
                    #medium_output = model.module.classify_medium(res['features'])
                    output = [
                        head_output + ratio*tail_output
                        for head_output in output
                    ]
                    """ output = [
                        tail_output
                        for head_output in output
                    ] """
                    
                    #import pdb; pdb.set_trace()
                    
                    s_output = output
                
            if return_features:
                if cali_mlp is not None:
                    features[ptr: ptr + bs] = fea
                else:
                    features[ptr: ptr+bs] = res['features']
                ptr += bs
            if return_input:
                inputs[ptr: ptr+bs] = images.reshape(bs, -1)
                ptr += bs

            if 'divclust' in cfg['name']:
                output = output[0]
                for index, output_ in enumerate(output):
                    for i, output_i in enumerate(output_):
                        #print(output_i.shape)
                        predictions[index].append(torch.argmax(output_i))
                        probs[index].append(output_i)
                        
            else:
                for i, output_i in enumerate(s_output):
                    predictions[i].append(torch.argmax(output_i, dim=1))
                    if cfg['method'] == 'cc' or 'cc' in cfg['name']:
                        probs[i].append(output_i)
                    else:
                        probs[i].append(F.softmax(output_i, dim=1))
                        
                for i, output_i in enumerate(output):
                    if cfg['method'] == 'cc' or 'cc' in cfg['name']:
                        output_val[i].append(output_i)
                    else:
                        output_val[i].append(output_i)
                    
            targets.append(targets_)
            
            """ if is_train:
                output_val.append(output[0]) """

    if 'divclust' not in cfg['name']:
        predictions = [torch.cat(pred_, dim = 0).cpu() for pred_ in predictions]
        probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
        output_val = [torch.cat(prob_, dim=0).cpu() for prob_ in output_val]
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.cat(targets, dim=0)
        return {'predictions': predictions, 'probs': probs, 'targets': targets, 'outputs':output_val}

    out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'outputs': out_} for pred_, prob_, out_ in zip(predictions, probs, output_val)]
    
    #print('out: ',len(out))
    
    if return_input:
        return out, inputs
    if return_features:
        return out, features
    elif is_train:
        output_val = torch.cat(output_val, dim=0)
        return out, output_val
    else:
        return out

from cdc.methods.dyn_train import SampleMasterTracker
@torch.no_grad()
def get_predictions_su(cfg, dataloader, model, tracker:SampleMasterTracker, return_features=False, is_train=False, cali_mlp = None, return_input=False, selector=None):
    # Make predictions on a dataset with neighbors
    model.eval()
    if cali_mlp is not None:
        cali_mlp.eval()
    predictions = [[] for _ in range(cfg['backbone']['nheads'])]
    probs = [[] for _ in range(cfg['backbone']['nheads'])]
    output_val = [[] for _ in range(cfg['backbone']['nheads'])]
    if 'divclust' in cfg['name']:
        predictions = [[] for _ in range(cfg['clusterings'])]
        probs = [[] for _ in range(cfg['clusterings'])]
    targets = []
    if return_features:
        if 'spicewide3' in cfg['name']:
            if 'cifar10' in cfg['name']:
                ft_dim = 128
            if 'cifar20' in cfg['name']:
                ft_dim = 512
            if 'stl10' in cfg['name']:
                ft_dim = 256
        else:
            ft_dim = get_feature_dimensions_backbone(cfg)
            
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()
    if return_input:
        s1,s2,s3,s4 = dataloader.dataset.data.shape
        inputs = torch.zeros((s1, s2*s3*s4)).cuda()
    ptr = 0
    with torch.no_grad():
        for batch in dataloader:
            if is_train:
                images = batch['image'].cuda(non_blocking=True)
                targets_ = batch['target'].cuda(non_blocking=True)
            else:
                #images, targets_ = batch[0], batch[1]
                images = batch['val'].cuda(non_blocking=True)
                targets_ = batch['target'].cuda(non_blocking=True)
                images_w = batch['image'].cuda(non_blocking=True)
                images_s = batch['image_augmented'].cuda(non_blocking=True)
                images_index = batch['index'].cuda(non_blocking=True)
                feature_weak = model(images_w, forward_pass='backbone')
                feature_augmented = model(images_s, forward_pass='backbone')
                # 计算特征稳定性指标
                feature_stability = F.cosine_similarity(feature_weak, feature_augmented, dim=1)
                stability_loss = 1 - feature_stability  # 转换为损失形式，越小表示越稳定

                indices= images_index

            bs = images.shape[0]
            
            #import pdb; pdb.set_trace()

            if cali_mlp is not None:
                fea = model(images.cuda(non_blocking=True),
                                     forward_pass='backbone')
                output = [cali_mlp(fea, forward_pass='calibration')]
                s_output = output

                
                
            elif 'divclust' in cfg['name']:
                res = model(images.cuda(non_blocking=True),
                        forward_pass='return_all')
                output = res['output']
                
            else:
                res = model(images.cuda(non_blocking=True),
                        forward_pass='return_all')
                output = res['output']
                s_output = res['output']
                
                if selector is not None:
                    selector.update(indices, output[0])
                
            if return_features:
                if cali_mlp is not None:
                    features[ptr: ptr + bs] = fea
                else:
                    features[ptr: ptr+bs] = res['features']
                ptr += bs
            if return_input:
                inputs[ptr: ptr+bs] = images.reshape(bs, -1)
                ptr += bs

            if 'divclust' in cfg['name']:
                output = output[0]
                for index, output_ in enumerate(output):
                    for i, output_i in enumerate(output_):
                        #print(output_i.shape)
                        predictions[index].append(torch.argmax(output_i))
                        probs[index].append(output_i)
                        
            else:
                for i, output_i in enumerate(s_output):
                    predictions[i].append(torch.argmax(output_i, dim=1))
                    if cfg['method'] == 'cc' or 'cc' in cfg['name']:
                        probs[i].append(output_i)
                    else:
                        probs[i].append(F.softmax(output_i, dim=1))
                        
                for i, output_i in enumerate(output):
                    if cfg['method'] == 'cc' or 'cc' in cfg['name']:
                        output_val[i].append(output_i)
                    else:
                        output_val[i].append(output_i)
                    
            targets.append(targets_)
            #pdb.set_trace()
            conf = probs[0][0].max(dim=1).values
            
            tracker.update(
                    indices=images_index.tolist(),
                    confidences=conf.tolist(),
                    labels=predictions[0][0].tolist(),
                    losses=stability_loss.tolist()
                )
    
    tracker.step()
    print(f"Removed={len(tracker.removed)}, restored={len(tracker.restore_log)}")
            
    if 'divclust' not in cfg['name']:
        predictions = [torch.cat(pred_, dim = 0).cpu() for pred_ in predictions]
        probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
        output_val = [torch.cat(prob_, dim=0).cpu() for prob_ in output_val]
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.cat(targets, dim=0)
        return {'predictions': predictions, 'probs': probs, 'targets': targets, 'outputs':output_val}

    out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'outputs': out_} for pred_, prob_, out_ in zip(predictions, probs, output_val)]
    
    if return_input:
        return out, inputs
    if return_features:
        return out, features
    elif is_train:
        output_val = torch.cat(output_val, dim=0)
        return out, output_val
    else:
        return out
    
@torch.no_grad()
def get_overlap(cfg, dataloader, model):
    # Make predictions on a dataset with neighbors
    model.eval()
    cur_overlap = torch.zeros(len(dataloader.dataset), cfg['backbone']['nclusters']).cuda()
    with torch.no_grad():
        for batch in dataloader:
            images, images_index = batch['val'], batch['index']
            bs = images.shape[0]
            res = model(images.cuda(non_blocking=True), forward_pass='return_all')
            output = res['output'][0]
            output_softmax = F.softmax(output, dim=1)

            pre_idx = torch.zeros(output.shape[0], output.shape[1]).cuda()
            for label_idx in range(output.shape[1]):
                sample_mask = output_softmax[:, label_idx].sort(descending=True)[1][:cfg['method_kwargs']['prototype_num']]
                pre_idx[sample_mask, label_idx] += 1
            cur_overlap[images_index] = pre_idx
    return cur_overlap

@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    # assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_acc_calibration_pdf(path, output_softmax, ground_truth, acc, ece,
                         n_bins = 15, title = None, epoch=None):
    p_value = np.max(output_softmax, 1)
    pred_label = np.argmax(output_softmax, 1)
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)

    sub_n_bins = n_bins * 3
    bins = np.arange(0, 1.0 + 1 / sub_n_bins-0.0001, 1 / sub_n_bins)
    sub_weights = np.ones(len(ground_truth)) / float(len(ground_truth))
    sub_acc = np.zeros_like(ground_truth, dtype=float)
    for index, value in enumerate(p_value):
        #value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) -0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]
    for index, value in enumerate(p_value):
        interval = int(value / (1 / n_bins) - 0.0001)
        sub_acc[index] = confidence_acc[interval]

    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    # plt.figure(figsize=(6, 5))
    plt.rcParams["font.weight"] = "bold"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.05, color='orange', label='Expected')
    ax.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
           alpha=0.9, width=0.05, color='dodgerblue', label='Outputs')



    # ax.set_aspect(1.)
    ax.plot([0,1], [0,1], ls='--',c='k')
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 0.6, pad=0.3, sharex=ax)
    ax_histy = divider.append_axes("right", 0.6, pad=0.3, sharey=ax)
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    # ax_histx.hist(p_value, bins=bins, edgecolor='white',
    #               color='lightblue', weights=sub_weights)
    # ax_histy.hist(sub_acc, bins=bins, orientation='horizontal',
    #               edgecolor='white', color='lightblue', weights=sub_weights)

    density = gaussian_kde(p_value)
    xs = np.linspace(0, 1, 30)
    density.covariance_factor = lambda: .05
    density._compute_covariance()
    f = interpolate.interp1d(xs, density(xs)/density(xs).sum(), kind='cubic')
    nx = np.linspace(0, 1, 100)
    ny = f(nx)
    ax_histx.plot(nx, ny,
                  c='dodgerblue', linewidth=3)
    density = gaussian_kde(sub_acc)
    xs = np.linspace(0, 1, 30)
    density.covariance_factor = lambda: .05
    density._compute_covariance()
    f = interpolate.interp1d(xs, density(xs)/density(xs).sum(), kind='cubic')
    nx = np.linspace(0, 1, 100)
    ny = f(nx)
    ax_histy.plot(ny, nx,
                  c='dodgerblue', linewidth=3)

    ax_histx.plot([p_value.mean().tolist(), p_value.mean().tolist()], [0, 1], ls='-', c='r', linewidth=3)
    ax_histy.plot([0, 1], [acc, acc], ls='-', c='r', linewidth=3)

    ax_histx.set_yticks([0, 0.5, 1])
    ax_histy.set_xticks([0, 0.5, 1])
    ax_histx.set_ybound(0,1)
    ax_histy.set_xbound(0,1)
    ax_histx.tick_params(labelsize=12)
    ax_histy.tick_params(labelsize=12)
    ax_histx.set_ylabel('% of Samples', fontsize=12, weight='bold')
    ax_histy.set_xlabel('% of Samples', fontsize=12, weight='bold')
    ax_histy.set_xticklabels(labels=[0,0.5,1.0],rotation=270)
    ax_histy.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    ax.set_xlabel('Confidence', fontsize=18, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=18, weight='bold')
    ax.tick_params(labelsize=16)
    ax.set_xbound(0, 1.0)
    ax.set_ybound(0, 1.0)

    if epoch is not None:
        plt.title(title+' Epoch: '+str(epoch), fontsize=18,
                  fontweight="bold", x=-4, y=1.37)
    # if title is not None:
    #     ax.set_title(title, fontsize=16, fontweight="bold", pad=-3)


    ax.legend(fontsize=18)

    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
    ax.text(0.95, 0.15,
            "ACC="+str(round(acc*100,1)) +"%"+'\n'+"Avg. Conf.="+str(round(p_value.mean()*100,1))+"%"+'\n'+"ECE="+str(round(ece*100,1))+"%",
            ha="right", va="center", size=16,
            bbox=bbox_props)
    # ax.text(0.95, 0.15,
    #         "ACC="+str(round(acc*100,1)) +"%"+'\n'+"ECE="+str(round(ece*100,1))+"%",
    #         ha="right", va="center", size=16,
    #         bbox=bbox_props)
    ax_histx.text(p_value.mean().tolist()-0.03, 0.5, "Avg.", rotation=90,
                  ha="center", va="center", size=16)
    ax_histy.text(0.5, acc-0.04, "Avg.",
                  ha="center", va="center", size=16)
    # plt.savefig(path+'/'+ title + '_epoch_' + str(epoch) +'.png', format='png', dpi=300,
    #             pad_inches=0, bbox_inches = 'tight')
    plt.savefig(path+'/'+ title + '_epoch_' + str(epoch) +'.pdf', format='pdf', dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    # plt.show()
    plt.close()
    
def plot_acc_calibration(path, output_softmax, ground_truth, acc, ece,
                         n_bins = 15, title = None, epoch=None):
    p_value = np.max(output_softmax, 1)
    pred_label = np.argmax(output_softmax, 1)
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)
    
    # 计算最大值、最小值、均值
    max_value = np.max(p_value)
    min_value = np.min(p_value)
    mean_value = np.mean(p_value)

    # 获取前10个值
    top_10_values = p_value[:10]

    # 输出结果
    print(f"最大值: {max_value}")
    print(f"最小值: {min_value}")
    print(f"均值: {mean_value}")
    #print(f"前10个值: {top_10_values}")
    
    sub_n_bins = n_bins * 3
    bins = np.arange(0, 1.0 + 1 / sub_n_bins-0.0001, 1 / sub_n_bins)
    sub_weights = np.ones(len(ground_truth)) / float(len(ground_truth))
    sub_acc = np.zeros_like(ground_truth, dtype=float)
    for index, value in enumerate(p_value):
        #value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) -0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
            
    
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]
            
    for index, value in enumerate(p_value):
        interval = int(value / (1 / n_bins) - 0.0001)
        sub_acc[index] = confidence_acc[interval]

    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    # plt.figure(figsize=(6, 5))
    plt.rcParams["font.weight"] = "bold"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # 绘制柱状图
    ax.bar(np.around(np.arange(start, 1.0, step), 3),
        np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.05, color='orange', label='Expected')
    ax.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
        alpha=0.9, width=0.05, color='dodgerblue', label='Outputs')

    # 绘制对角线
    ax.plot([0, 1], [0, 1], ls='--', c='k')

    # 设置坐标轴和标签
    ax.set_xlabel('Confidence', fontsize=18, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=18, weight='bold')
    ax.tick_params(labelsize=16)
    ax.set_xbound(0, 1.0)
    ax.set_ybound(0, 1.0)

    # 设置图例
    ax.legend(fontsize=18)

    # 添加文本框
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
    ax.text(0.95, 0.15,
            "ACC=" + str(round(acc * 100, 1)) + "%" + '\n' +
            "Avg. Conf.=" + str(round(p_value.mean() * 100, 1)) + "%" + '\n' +
            "ECE=" + str(round(ece * 100, 1)) + "%",
            ha="right", va="center", size=16, bbox=bbox_props)

    # 调整标题位置
    if epoch is not None:
        plt.title(title + ' Epoch: ' + str(epoch), fontsize=18,
                fontweight="bold", pad=20)

    # 调整布局，确保内容居中
    plt.tight_layout()

    # 保存图像
    plt.savefig(path + title + '_epoch_' + str(epoch) + '.png',
                format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)

    # plt.show()  # 如果需要显示图像
    plt.close()

@torch.no_grad()
def hungarian_evaluate(cfg: object, path: object, epoch: object, subhead_index: object, all_predictions: object,
                       title: object, class_names: object = None,
                       compute_confusion_matrix: object = True, confusion_matrix_file: object = None,
                       features: object = None, save_wrong: object = False) -> object:
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.
    # print("checkpoint 3")
    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    outputs = head['outputs'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    #print(probs[0])
    #print(outputs[0])
    #print('*******')
    #print(predictions.shape, targets.shape)

    match = _hungarian_match(predictions, targets, preds_k=probs.shape[1], targets_k=num_classes)
    
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    
    reordered_probs = torch.zeros((num_elems,num_classes), dtype=probs.dtype).cuda()
    #reordered_outputs = torch.zeros((num_elems,num_classes), dtype=outputs.dtype).cuda()
    
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
        reordered_probs[:,target_i]=probs[:,pred_i]
        #reordered_outputs[:,target_i]=outputs[:,pred_i]
    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    if save_wrong:
        error_mask = (reordered_preds != targets)                 # 布尔掩码
        error_indices = torch.nonzero(error_mask).squeeze(1)  # [num_errors]

        # 2. 计算每个样本的“预测置信度”：取模型给出的最大概率
        confidences, pred_labels = reordered_probs.max(dim=1)           # [num_elems]

        #import pdb; pdb.set_trace()

        # 3. 仅保留错误样本的置信度
        error_confidences = confidences[error_indices]        # [num_errors]

        # 4. 根据置信度从大到小，对错误样本索引排序
        sorted_order = torch.argsort(error_confidences, descending=True)
        sorted_error_indices = error_indices[sorted_order]    # [num_errors]，已按置信度降序

        # 5. 保存到文件（如 CSV 或 TXT），这里以 TXT 为例，每行一个索引
        out_path = os.path.join(path, 'sorted_misclassified_indices.txt')
        np.savetxt(out_path, sorted_error_indices.cpu().numpy(), fmt='%d')
        print(f"已将 {sorted_error_indices.numel()} 个错误样本索引（按置信度降序）保存到：{out_path}")
        
    if features is None:
        sc, chi, dbi, dunn_index, dbcv_score = 0, 0, 0, 0, 0
    else:
        sc, chi, dbi, dunn_index, dbcv_score = 0, 0, 0, 0, 0
        # sc = metrics.silhouette_score(features.cpu().numpy(), predictions.cpu().numpy())
        # chi = metrics.calinski_harabasz_score(features.cpu().numpy(), predictions.cpu().numpy())
        # dbi = metrics.davies_bouldin_score(features.cpu().numpy(), predictions.cpu().numpy())
        # dunn_index = dunn(features.cpu().numpy(), predictions.cpu().numpy())
        #
        # clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
        # clusterer.fit(features.cpu().numpy())
        # dbcv_score = clusterer.relative_validity_
    #############
    # https://github.com/Impression2805/FMFP/blob/main/utils/metrics.py#L142
    AURC, EAURC = calc_aurc_eaurc(probs.cpu().tolist(), (reordered_preds == targets).cpu().tolist())
    
    AUROC, AUPR_success, AUPR_err, FPR95, TNR95 = calc_fpr_aupr(outputs.cpu().tolist(),(reordered_preds == targets).cpu().tolist())
    """ print(f"logits: AUROC: {AUROC:.4f}, AUPR_success: {AUPR_success:.4f}, AUPR_err: {AUPR_err:.4f}, FPR95: {FPR95:.4f}, TNR95: {TNR95:.4f}") """
    
    AUROC, AUPR_success, AUPR_err, FPR95, TNR95 = calc_fpr_aupr(probs.cpu().tolist(),(reordered_preds == targets).cpu().tolist())
    """ print(f"softmax: AUROC: {AUROC:.4f}, AUPR_success: {AUPR_success:.4f}, AUPR_err: {AUPR_err:.4f}, FPR95: {FPR95:.4f}, TNR95: {TNR95:.4f}") """

    ############
    #_, preds_top5 = probs.topk(5, 1, largest=True)
    _, preds_top5 = probs.topk(min(5,num_classes), 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)
    ece = calibration_error(reordered_probs, targets, task='multiclass', num_classes=num_classes, n_bins=15).item()
    conf = reordered_probs.max(1)[0].mean().item()

    dist = reordered_preds.unique(return_counts=True)[1]
    imb_ratio = (dist.max()/dist.min()).item()

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(), 
                            class_names, confusion_matrix_file)
    # print("checkpoint 4")
    # cal calibration
    """ if (epoch == 0) or (epoch+1) % cfg['cluster_eval']['plot_freq'] == 0:
        try:
            plot_acc_calibration(path, reordered_probs.cpu().numpy(),
                             targets.cpu().numpy(), acc, ece,
                             n_bins = 15, title= title,
                             epoch=epoch+1)
            # print("checkpoint 5")
        except:
            pass """
    if epoch ==998:
        wandb.log({
            'TACC': round(acc*100,6),'TNMI': round(nmi*100,6),
            'TARI': round(ari*100,6), 'TECE': round(ece*100, 6), 'CONF': round(conf*100,6),
            'IMBRATO':round(imb_ratio,6),
            'TACC Top-5': round(top5*100,6),
            'AURC': round(AURC*100,6), 'EAURC': round(EAURC*100,6), 'AUROC': round(AUROC*100,6),
            'AUPR_success': round(AUPR_success*100,6), 'AUPR_err': round(AUPR_err*100,6),
            'FPR95': round(FPR95*100,6), 'TNR95': round(TNR95*100,6),

        'T_SC': round(sc, 6), 'T_CHI': round(chi, 6), 'T_DBI': round(dbi, 6)
        })
    else:
        wandb.log({
            'epoch': epoch,
            'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
            'ARI': round(ari*100, 6), 'ECE': round(ece*100, 6), 'CONF': round(conf*100,6),
            'IMBRATO': round(imb_ratio, 6),
            'ACC Top-5': round(top5*100, 6),
            'AURC': round(AURC*100,6), 'EAURC': round(EAURC*100,6), 'AUROC': round(AUROC*100,6),
            'AUPR_success': round(AUPR_success*100,6), 'AUPR_err': round(AUPR_err*100,6),
            'FPR95': round(FPR95*100,6), 'TNR95': round(TNR95*100,6),

            'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6)
        })

    return {'ACC': round(acc*100,6),'NMI': round(nmi*100,6),
            'ARI': round(ari*100,6), 'ECE': round(ece*100,6), 'CONF': round(conf*100,6),
            'IMBRATO': round(imb_ratio, 6),
            'ACC Top-5': round(top5*100,6),

            'AURC': round(AURC*100,6), 'EAURC': round(EAURC*100,6), 'AUROC': round(AUROC*100,6),
            'AUPR_success': round(AUPR_success*100,6), 'AUPR_err': round(AUPR_err*100,6),
            'FPR95': round(FPR95*100,6), 'TNR95': round(TNR95*100,6),

            'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6),

            'hungarian_match': match}
    
@torch.no_grad()
def hungarian_evaluate_hard(cfg: object, path: object, epoch: object, subhead_index: object, all_predictions: object,
                    title: object, class_names: object = None,
                    compute_confusion_matrix: object = True, confusion_matrix_file: object = None,
                    features: object = None, _indices = None, easy=True, bins: list=None) -> object:
    
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.
    # print("checkpoint 3")
    # Hungarian matching
    
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    outputs = head['outputs'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)
    
    #import pdb; pdb.set_trace()

    match = _hungarian_match(predictions, targets, preds_k=probs.shape[1], targets_k=num_classes)
    
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    
    reordered_probs = torch.zeros((num_elems, num_classes), dtype=probs.dtype).cuda()
    
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
        reordered_probs[:,target_i]=probs[:,pred_i]

    #import pdb; pdb.set_trace()
    
    uniform_target = torch.full_like(outputs, 1.0 / num_classes).cuda()
    sample_losses = F.mse_loss(outputs, uniform_target, reduction='none')     
    sample_losses = sample_losses.mean(dim=1)
    hard_vector = []
    hard_vector.extend(sample_losses.tolist())

    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    if _indices is None:
        if easy:
            #indices = get_top_indices_balanced(hard_vector, reordered_preds.cpu().numpy(), p=0.01, balance=True)
            indices = get_indices_in_range_balanced(hard_vector, reordered_preds.cpu().numpy(), p_high=0.95, p_low=0.05, balance=True)
            print(f"Indices of top 5% values: {len(indices), indices[:20]}")
            
        else:
            indices = get_bottom_indices_balanced(hard_vector, reordered_preds.cpu().numpy(), p=0.6, balance=True)
            print(f"Indices of bottom values: {len(indices), indices[:20]}")
        
    else:
        indices = _indices
        
    #import pdb; pdb.set_trace()
    print("Evaluating hard samples……")
    labels_hard = targets.cpu().numpy()[indices]
    predictions_hard = predictions.cpu().numpy()[indices]
    pred_hard = reordered_preds.cpu().numpy()[indices]
    
    acc_h = np.sum(pred_hard == labels_hard) / len(labels_hard)
    nmi_h = metrics.normalized_mutual_info_score(labels_hard, predictions_hard)
    ari_h = metrics.adjusted_rand_score(labels_hard, predictions_hard)
    print("Hard ACC, NMI, ARI:", acc_h*100, nmi_h*100, ari_h*100)
    #import pdb; pdb.set_trace()
    if features is None:
        sc, chi, dbi, dunn_index, dbcv_score = 0, 0, 0, 0, 0
    else:
        sc, chi, dbi, dunn_index, dbcv_score = 0, 0, 0, 0, 0

    if bins is not None:
        for i in range(len(bins)):
            #continue
            bin_ = torch.tensor(bins[i].tolist(), device=targets.device, dtype=torch.long)
            if len(bin_)>0:
                bin_preds = reordered_preds[bin_]
                bin_targets = targets[bin_]
                bin_acc = (bin_preds == bin_targets).float().mean().item()
                print(f"bin acc group{i}: {bin_acc:.4f}, bin size: {len(bin_)} ")
                log_path = os.path.join(cfg['cdc_dir'], 'training_log.log')
                log_file = open(log_path, 'a')
                log_file.write(f"bin accgroup{i}: {bin_acc:.4f}, bin size: {len(bin_)}\n ")
                log_file.close() 
        
    AURC, EAURC = calc_aurc_eaurc(probs.cpu().tolist(), (reordered_preds == targets).cpu().tolist())
    
    AUROC, AUPR_success, AUPR_err, FPR95, TNR95 = calc_fpr_aupr(outputs.cpu().tolist(),(reordered_preds == targets).cpu().tolist())
    """ print(f"logits: AUROC: {AUROC:.4f}, AUPR_success: {AUPR_success:.4f}, AUPR_err: {AUPR_err:.4f}, FPR95: {FPR95:.4f}, TNR95: {TNR95:.4f}") """
    
    AUROC, AUPR_success, AUPR_err, FPR95, TNR95 = calc_fpr_aupr(probs.cpu().tolist(),(reordered_preds == targets).cpu().tolist())
    """ print(f"softmax: AUROC: {AUROC:.4f}, AUPR_success: {AUPR_success:.4f}, AUPR_err: {AUPR_err:.4f}, FPR95: {FPR95:.4f}, TNR95: {TNR95:.4f}") """

    ############
    #_, preds_top5 = probs.topk(5, 1, largest=True)
    _, preds_top5 = probs.topk(min(5,num_classes), 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)
    ece = calibration_error(reordered_probs, targets, task='multiclass', num_classes=num_classes, n_bins=15).item()
    conf = reordered_probs.max(1)[0].mean().item()

    dist = reordered_preds.unique(return_counts=True)[1]
    imb_ratio = (dist.max()/dist.min()).item()
    
    """ if epoch >= 0 and epoch%10==0 and features is not None:
        visualize_tsne(features, reordered_preds.cpu().numpy(), indices, path, epoch, num_classes) """

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(), 
                            class_names, confusion_matrix_file)
    # cal calibration
    """ if (epoch == 0) or (epoch+1) % cfg['cluster_eval']['plot_freq'] == 0:
        try:
            plot_acc_calibration(path, reordered_probs.cpu().numpy(),
                             targets.cpu().numpy(), acc, ece,
                             n_bins = 15, title= title,
                             epoch=epoch+1)
            # print("checkpoint 5")
        except:
            pass """
    if epoch ==998:
        wandb.log({
            'TACC': round(acc*100,6),'TNMI': round(nmi*100,6),
            'TARI': round(ari*100,6), 'TECE': round(ece*100, 6), 'CONF': round(conf*100,6),
            'IMBRATO':round(imb_ratio,6),
            'TACC Top-5': round(top5*100,6),
            'AURC': round(AURC*100,6), 'EAURC': round(EAURC*100,6), 'AUROC': round(AUROC*100,6),
            'AUPR_success': round(AUPR_success*100,6), 'AUPR_err': round(AUPR_err*100,6),
            'FPR95': round(FPR95*100,6), 'TNR95': round(TNR95*100,6),

        'T_SC': round(sc, 6), 'T_CHI': round(chi, 6), 'T_DBI': round(dbi, 6)
        })
        
    else:
        wandb.log({
            'epoch': epoch,
            'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
            'ARI': round(ari*100, 6), 'ECE': round(ece*100, 6), 'CONF': round(conf*100,6),
            'IMBRATO': round(imb_ratio, 6),
            'ACC Top-5': round(top5*100, 6),
            'AURC': round(AURC*100,6), 'EAURC': round(EAURC*100,6), 'AUROC': round(AUROC*100,6),
            'AUPR_success': round(AUPR_success*100,6), 'AUPR_err': round(AUPR_err*100,6),
            'FPR95': round(FPR95*100,6), 'TNR95': round(TNR95*100,6),

            'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6)
        })

    return {'ACC': round(acc*100,6),'NMI': round(nmi*100,6),
            'ARI': round(ari*100,6), 'ECE': round(ece*100,6), 'CONF': round(conf*100,6),
            'IMBRATO': round(imb_ratio, 6),
            'ACC Top-5': round(top5*100,6),

            'AURC': round(AURC*100,6), 'EAURC': round(EAURC*100,6), 'AUROC': round(AUROC*100,6),
            'AUPR_success': round(AUPR_success*100,6), 'AUPR_err': round(AUPR_err*100,6),
            'FPR95': round(FPR95*100,6), 'TNR95': round(TNR95*100,6),

            'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6),

            'hungarian_match': match}, indices

@torch.no_grad()
def generate_and_plot_tsne(cfg, dataloader, model, indices_to_highlight=None, save_path="tsne.png", epoch=0, draw=True, neighbors=False):
    """
    从模型提取特征，生成 t-SNE 并绘图（保存到文件）。
    参数：
    - cfg: 配置文件字典
    - dataloader: 数据加载器
    - model: 训练好的模型
    - indices_to_highlight: 需要特别标注的样本索引（List[int]）
    - save_path: 图片保存路径
    """
    # Step 1: 提取特征和标签
    out, features = get_predictions(cfg, dataloader, model, return_features=True)
    targets = out[0]['targets']
    predictions = out[0]['predictions']  # 使用预测类别
    confidences = out[0]['probabilities']
    predictions_np = predictions.detach().cpu().numpy()
    
    confidences_np = confidences.detach().cpu().numpy()
    max_conf = np.max(confidences_np, axis=1)   # 每个样本的最大置信度值

    from collections import defaultdict, Counter
    conf_per_class = defaultdict(list)

    for pred, conf in zip(predictions_np, max_conf):
        conf_per_class[pred].append(conf)

    print("\n每个预测类别的置信度统计:")
    for cls in sorted(conf_per_class.keys()):
        cls_confs = conf_per_class[cls]
        print(f"  类别 {cls}: 平均置信度 = {np.mean(cls_confs):.4f}, 标准差 = {np.std(cls_confs):.4f}, 样本数 = {len(cls_confs)}")
        
    class_counts = Counter(predictions_np)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    total_classes = len(sorted_classes)
    num_head = int(total_classes * 0.3)
    num_tail = int(total_classes * 0.3)

    head_classes = set(cls for cls, _ in sorted_classes[:num_head])
    tail_classes = set(cls for cls, _ in sorted_classes[-num_tail:])
    middle_classes = set(cls for cls, _ in sorted_classes[num_head:-num_tail])

    # Step 1: 统计每个子集的样本数
    print("\n每个子集的样本数:")
    print(f"  Head classes ({len(head_classes)}): {head_classes}")
    print(f"  Middle classes ({len(middle_classes)}): {middle_classes}")
    print(f"  Tail classes ({len(tail_classes)}): {tail_classes}")

    
    
    
    # Step 2: 转为 numpy
    features_np = features.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    #targets_np = predictions_np  # 使用预测类别作为标签
    
    """ def re_cluster_within_class(features_np, predictions_np, targets_np, target_class, save_path, epoch=0, n_subclusters=2):
        # Step 1: 取出该类的特征及其真实标签
        class_mask = (predictions_np == target_class)
        class_feats = features_np[class_mask]
        class_targets = targets_np[class_mask]

        if len(class_feats) < n_subclusters:
            print(f"⚠️ 类别 {target_class} 样本数过少，无法分为 {n_subclusters} 个簇")
            return None, class_mask

        # Step 2: PCA 降维到 2D
        pca = PCA(n_components=2)
        class_feats_2d = pca.fit_transform(class_feats)

        # Step 3: KMeans 二次聚类
        kmeans = KMeans(n_clusters=n_subclusters, random_state=42)
        subcluster_labels = kmeans.fit_predict(class_feats)

        # Step 4: 可视化真实标签 vs 子聚类标签
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # 左图：真实标签分布
        unique_targets = np.unique(class_targets)
        for label in unique_targets:
            idxs = class_targets == label
            axs[0].scatter(class_feats_2d[idxs, 0], class_feats_2d[idxs, 1],
                        label=f'True-{label}', s=10, alpha=0.7)
        axs[0].set_title(f"GT labels in predicted class {target_class}")
        axs[0].legend()
        axs[0].grid(True)

        # 右图：KMeans 子聚类结果
        for sub_id in range(n_subclusters):
            idxs = subcluster_labels == sub_id
            axs[1].scatter(class_feats_2d[idxs, 0], class_feats_2d[idxs, 1],
                        label=f'KMeans-{sub_id}', s=10, alpha=0.7)
        axs[1].set_title(f"KMeans sub-clustering of class {target_class}")
        axs[1].legend()
        axs[1].grid(True)

        plt.suptitle(f"Sub-cluster vs GT for predicted class {target_class}")
        plt.tight_layout()

        # Step 5: 保存图片
        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, f"subcluster_vs_gt_epoch{epoch}_class{target_class}.png")
        plt.savefig(full_path, dpi=300)
        plt.close()
        print(f"✅ Sub-clustering 对比图已保存到：{full_path}")

        return subcluster_labels, class_mask """
    
    # 提取类别 10 的样本再做一次 KMeans，看看是否能拆分为两个稳定子簇
    """ re_cluster_within_class(features_np, predictions_np, targets_np, target_class=1, epoch=epoch, save_path=save_path, n_subclusters=2)
    re_cluster_within_class(features_np, predictions_np, targets_np, target_class=4, epoch=epoch, save_path=save_path, n_subclusters=2) """
    
    if draw: 
        if indices_to_highlight is not None and len(indices_to_highlight) > 0:
            indices_to_highlight = np.array(indices_to_highlight)

            # 获取这些 index 对应的预测类别
            highlighted_preds = predictions_np[indices_to_highlight]

            # 统计每个预测类别中有多少被高亮
            from collections import Counter
            highlight_counter = Counter(highlighted_preds)

            print("每个预测类别中，highlighted 样本数量如下：")
            for cls in sorted(highlight_counter):
                print(f"  类别 {cls}: {highlight_counter[cls]} 个样本")
                

        print("PCA……")
        pca = PCA(n_components=50, random_state=42)
        features_pca = pca.fit_transform(features_np)

        # Step 5: t-SNE 降到二维（设置perplexity与迭代次数）
        print("TSNE……")
        tsne = TSNE(n_components=2, perplexity=40, n_iter=500, init='pca', random_state=42)
        print("Fit……")
        features_2d = tsne.fit_transform(features_pca)

        """ # Step 4: 绘图
        plt.figure(figsize=(10, 8))
        num_classes = len(np.unique(targets_np))
        colors = plt.cm.get_cmap('tab20', num_classes)

        for class_idx in range(num_classes):
            idxs = targets_np == class_idx
            plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1],
                        label=f"Class {class_idx}", alpha=0.6, s=15, color=colors(class_idx))

        if indices_to_highlight is not None and len(indices_to_highlight) > 0:
            indices_to_highlight = np.array(indices_to_highlight)
            plt.scatter(features_2d[indices_to_highlight, 0],
                        features_2d[indices_to_highlight, 1],
                        facecolors='none', edgecolors='red',
                        s=80, linewidths=1.5, label='Highlighted')

        plt.legend()
        plt.title("t-SNE of SCAN Features")
        save_path1 = os.path.join(save_path, f"tsne_plot_epoch_{epoch}.png")
        plt.savefig(save_path1, dpi=300)
        plt.close()
        print(f"t-SNE 图已保存到 {save_path}") """
        
        plt.figure(figsize=(10, 8))
        num_classes = len(np.unique(targets_np))
        colors = plt.cm.get_cmap('tab20', num_classes)

        for class_idx in range(num_classes):
            idxs = targets_np == class_idx
            plt.scatter(features_2d[idxs, 0], features_2d[idxs, 1],
                        label=f"Class {class_idx}", alpha=0.6, s=15, color=colors(class_idx))

        # 高亮指定样本
        if indices_to_highlight is not None and len(indices_to_highlight) > 0:
            indices_to_highlight = np.array(indices_to_highlight)
            plt.scatter(features_2d[indices_to_highlight, 0],
                        features_2d[indices_to_highlight, 1],
                        facecolors='none', edgecolors='red',
                        s=80, linewidths=1.5, label='Highlighted')

        # ⭐ 每个簇的中心点
        print("计算每个预测簇的中心点并绘制⭐...")
        cluster_ids = predictions_np
        unique_clusters = np.unique(cluster_ids)
        for cluster_id in unique_clusters:
            cluster_indices = np.where(cluster_ids == cluster_id)[0]
            cluster_features_2d = features_2d[cluster_indices]
            cluster_center = np.mean(cluster_features_2d, axis=0)

            plt.scatter(cluster_center[0], cluster_center[1],
                        marker='*', color='black', s=200, edgecolors='white',
                        linewidths=1.5, label=f'Cluster {cluster_id} Center' if cluster_id == unique_clusters[0] else None)

        plt.legend()
        plt.title("t-SNE of SCAN Features (with Cluster Centers ⭐)")
        save_path1 = os.path.join(save_path, f"tsne_plot_epoch_{epoch}.png")
        plt.savefig(save_path1, dpi=300)
        plt.close()
        print(f"t-SNE 图已保存到 {save_path1}")
        
        cluster_ids = predictions_np  # 这里我们假设 targets_np 是 cluster_labels

        plot_confidence_distribution_by_cluster(
            confidences_np=confidences_np,
            cluster_labels=cluster_ids,
            true_labels=targets_np,
            save_path=save_path,
            epoch=epoch
        )
    
    if neighbors:
        return head_classes, middle_classes, tail_classes
    
def compute_features_and_neighbors(model, train_loader, train_dataset, args, pseudo_labels):
    """
    提取特征 + 计算每个样本的类别近邻（head/middle/tail），用于 long-tail aware SCAN 训练。

    参数:
    - model: 已训练好的模型，支持 return_features=True
    - train_loader: DataLoader，需返回 (inputs, targets, indices)
    - train_dataset: 总训练集，用于分配 feature 空间
    - args: 参数，包含 args.num_class, args.k 等
    - pseudo_labels: 每个样本的伪标签 (N,) numpy array

    返回:
    - head_neighbors_idx: [N, ≤k]
    - medium_neighbors_idx: [N, ≤k]
    - tail_neighbors_idx: [N, ≤k]
    - head_classes: Set[int]
    - medium_classes: Set[int]
    - tail_classes: Set[int]
    """
    model.eval()
    num_samples = len(train_dataset)
    feature_dim = args['backbone']['feat_dim']  # 特征维度

    total_features = torch.zeros((num_samples, feature_dim)).cuda()
    total_labels = torch.zeros(num_samples).long().cuda()

    with torch.no_grad():
        for batch_id, output in enumerate(train_loader):
            inputs = output['image'] 
            indices = output['index']
            targets = output['target']
            
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs, forward_pass='return_all')
            feats = outputs['features']  
            total_features[indices] = feats
            total_labels[indices] = targets  # targets 可选，用于对比；也可以省略

    # -----------------------------
    # Step 1: 计算每类的特征中心（基于伪标签）
    # -----------------------------
    class_features = []
    for c in range(args['backbone']['nclusters']):
        idx = (torch.tensor(pseudo_labels).cuda() == c).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            class_center = total_features[idx].mean(dim=0)
        else:
            class_center = torch.zeros(feature_dim).cuda()
        class_features.append(class_center)
    class_features = torch.stack(class_features, dim=0)

    # -----------------------------
    # Step 2: 归一化 + 用 FAISS 查找每个样本的 k 个最近 class neighbor
    # -----------------------------
    total_features_np = total_features.cpu().numpy()
    class_features_np = class_features.cpu().numpy()

    total_features_np = total_features_np / np.linalg.norm(total_features_np, axis=1, keepdims=True)
    class_features_np = class_features_np / np.linalg.norm(class_features_np, axis=1, keepdims=True)

    index = faiss.IndexFlatIP(class_features_np.shape[1])  # cosine similarity = dot after norm
    index.add(class_features_np)

    k = int(args['backbone']['nclusters']*0.5)
    _, nearest_neighbors_idx = index.search(total_features_np, k)  # shape: [N, k]

    # -----------------------------
    # Step 3: 按类别样本数划分 head/mid/tail 类簇
    # -----------------------------
    pseudo_labels = pseudo_labels.cpu().tolist()
    class_counts = Counter(pseudo_labels)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    num_head = int(args['backbone']['nclusters'] * 0.3)
    num_tail = int(args['backbone']['nclusters'] * 0.3)

    head_classes = set(cls for cls, _ in sorted_classes[:num_head])
    tail_classes = set(cls for cls, _ in sorted_classes[-num_tail:])
    middle_classes = set(cls for cls, _ in sorted_classes[num_head:-num_tail])

    """ print("\n每个子集的样本数:")
    print(f"  Head classes ({len(head_classes)}): {head_classes}")
    print(f"  Middle classes ({len(middle_classes)}): {middle_classes}")
    print(f"  Tail classes ({len(tail_classes)}): {tail_classes}") """

    

    # -----------------------------
    # Step 4: 对每个样本的近邻类别做划分（head/mid/tail）
    # -----------------------------
    head_neighbors_idx = []
    medium_neighbors_idx = []
    tail_neighbors_idx = []

    for i in range(nearest_neighbors_idx.shape[0]):
        current_neighbors = nearest_neighbors_idx[i]

        head_neighbors = [cls for cls in current_neighbors if cls in head_classes]
        medium_neighbors = [cls for cls in current_neighbors if cls in middle_classes]
        tail_neighbors = [cls for cls in current_neighbors if cls in tail_classes]

        head_neighbors_idx.append(head_neighbors)
        medium_neighbors_idx.append(medium_neighbors)
        tail_neighbors_idx.append(tail_neighbors)

    return (
        np.array(head_neighbors_idx, dtype=object),
        np.array(medium_neighbors_idx, dtype=object),
        np.array(tail_neighbors_idx, dtype=object),
    )
    
def kmeans_cluster_and_visualize_by_true_label(class_list, features, dataset, n_clusters=2, save_path=None, epoch=0): 
    """
    对指定真实类别执行 KMeans 聚类，并将所有样本画在一张图上（用真实标签着色），打印聚类指标。
    """
    targets = np.array(dataset.targets)
    results = {}

    all_cls_features = []
    all_cls_labels = []
    all_cls_indices = []
    all_kmeans_labels = []

    for cls in class_list:
        indices = np.where(targets == cls)[0]
        cls_features = features[indices]
        if isinstance(cls_features, torch.Tensor):
            cls_features = cls_features.cpu().numpy()

        if len(indices) < n_clusters:
            print(f"[跳过] 类别 {cls} 样本不足 {n_clusters} 个，只有 {len(indices)}")
            continue

        # 聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(cls_features)

        # 聚类评估指标
        silhouette = silhouette_score(cls_features, cluster_labels)
        ch_score = calinski_harabasz_score(cls_features, cluster_labels)

        print(f"\n=== 类别 {cls} 聚类结果 ===")
        print(f"样本数量: {len(indices)}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Calinski-Harabasz Index: {ch_score:.4f}")

        results[cls] = {
            "indices": indices,
            "cluster_labels": cluster_labels,
            "silhouette": silhouette,
            "ch_index": ch_score
        }

        all_cls_features.append(cls_features)
        all_cls_labels.append(np.full(len(indices), cls))
        all_cls_indices.append(indices)
        all_kmeans_labels.append(cluster_labels)

    # 所有类数据合并
    if len(all_cls_features) == 0:
        print("没有足够样本的类别可用于可视化")
        return results
    
    global_cluster_labels = []
    for i, cluster_labels in enumerate(all_kmeans_labels):
        # 为每类分配不同的标签区间（避免重复）
        offset = i * n_clusters
        new_labels = cluster_labels + offset
        global_cluster_labels.extend(new_labels)
    
    all_features = np.concatenate(all_cls_features, axis=0)
    all_labels = np.concatenate(all_cls_labels, axis=0)
    pseudo_labels = np.concatenate(all_kmeans_labels)
    
    plot_tsne_true_vs_pseudo(all_features, all_labels, pseudo_labels, save_path=save_path, epoch=epoch)

    # ✅ 类间 Silhouette Score（真实标签）
    sil_score_true_labels = silhouette_score(all_features, all_labels)
    print(f"\n=== 多类结构评估 ===")
    print(f"Silhouette Score (Inter-class separation with true labels): {sil_score_true_labels:.4f}")

    # 可视化所有样本：t-SNE + 真实标签着色
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    tsne_embeds = tsne.fit_transform(all_features)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_embeds[:, 0], y=tsne_embeds[:, 1],
                    hue=all_labels, palette='tab10', s=40)
    plt.title(f"t-SNE of Selected Classes | Epoch {epoch}")
    plt.axis('off')
    plt.legend(title="True Class", loc="best")

    if save_path:
        plt.savefig(f"{save_path}/kmeans_tsne_classes{class_list}_epoch{epoch}.png", bbox_inches='tight', dpi=300)
    else:
        plt.show()

    # === 基于聚类中心距离的“置信度”估计 ===
    print("\n=== 计算基于 KMeans 的置信度分布 ===")
    all_confidences = []

    start_idx = 0
    for cls in class_list:
        if cls not in results:
            continue
        cls_feats = all_cls_features[start_idx]
        cluster_labels = results[cls]["cluster_labels"]

        # 用 kmeans 拿到聚类中心
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        kmeans.fit(cls_feats)
        centers = kmeans.cluster_centers_

        # 对每个样本，计算它到其聚类中心的距离
        for i in range(len(cls_feats)):
            label = cluster_labels[i]
            dist = np.linalg.norm(cls_feats[i] - centers[label])
            conf = 1.0 / (1.0 + dist)  # 距离越小 → 置信度越高
            all_confidences.append(conf)

        start_idx += 1

    # 绘图：置信度直方图
    plt.figure(figsize=(6, 4))
    plt.hist(all_confidences, bins=50, color='lightgreen', edgecolor='black')
    plt.title("KMeans-based Confidence Distribution")
    plt.xlabel("Confidence (1 / (1 + distance to cluster center))")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    if save_path:
        plt.savefig(f"{save_path}/kmeans_confidence_distribution_epoch{epoch}.png", bbox_inches='tight', dpi=300)
    else:
        plt.show()

    return results

def plot_tsne_true_vs_pseudo(features, true_labels, pseudo_labels, save_path=None, epoch=0):
    """
    绘制 t-SNE 图：真实标签 vs KMeans 伪标签（聚类结果）

    参数：
    - features: N x D numpy array，特征向量
    - true_labels: N 长度 array，真实标签（如 0/1）
    - pseudo_labels: N 长度 array，聚类标签（如 KMeans 子簇）
    """
    tsne = TSNE(n_components=2, random_state=42, init='pca')
    tsne_embeds = tsne.fit_transform(features)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # 图1：真实标签
    sns.scatterplot(x=tsne_embeds[:, 0], y=tsne_embeds[:, 1],
                    hue=true_labels, palette='tab10', s=30, ax=axs[0])
    axs[0].set_title(f"True Labels | Epoch {epoch}")
    axs[0].axis('off')
    axs[0].legend(title="True Label")

    # 图2：伪标签（KMeans 聚类标签）
    sns.scatterplot(x=tsne_embeds[:, 0], y=tsne_embeds[:, 1],
                    hue=pseudo_labels, palette='tab20', s=30, ax=axs[1])
    axs[1].set_title(f"KMeans Pseudo Labels | Epoch {epoch}")
    axs[1].axis('off')
    axs[1].legend(title="Cluster ID")

    plt.tight_layout()

    if save_path:
        plt.savefig(f"{save_path}/tsne_true_vs_kmeans_epoch{epoch}.png", bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_confidence_distribution_by_cluster(confidences_np, cluster_labels, true_labels,
                                            save_path, epoch=0):
    """
    为每个聚类簇绘制 max confidence 分布图，并标注真实类别构成。

    参数:
    - confidences_np: N x C array，softmax置信度
    - cluster_labels: N array，KMeans的伪标签（cluster id）
    - true_labels: N array，真实标签（如 ground-truth class）
    - save_path: 输出路径
    - epoch: 当前 epoch（用于保存文件名）
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np
    import math
    from collections import Counter

    max_conf = np.max(confidences_np, axis=1)
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    cols = 3
    rows = math.ceil(n_clusters / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, cid in enumerate(unique_clusters):
        row, col = divmod(idx, cols)
        ax = axs[row][col]
        mask = (cluster_labels == cid)

        # 绘制置信度直方图
        sns.histplot(max_conf[mask], bins=30, kde=False, ax=ax, color="skyblue", edgecolor="black")

        # 统计真实标签
        cluster_true_labels = true_labels[mask]
        label_counts = Counter(cluster_true_labels)
        sorted_counts = sorted(label_counts.items(), key=lambda x: -x[1])
        label_info = ", ".join([f"{lbl}:{cnt}" for lbl, cnt in sorted_counts[:3]])  # 只显示最多的3类

        ax.set_title(f"Cluster {cid}\nTrue: {label_info}")
        ax.set_xlabel("Max Confidence")
        ax.set_ylabel("Num Samples")

    # 清理空图
    for idx in range(n_clusters, rows * cols):
        fig.delaxes(axs[idx // cols][idx % cols])

    fig.suptitle(f"Confidence Histogram by Cluster | Epoch {epoch}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"conf_hist_true_label_epoch_{epoch}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✅ 每簇置信度直方图（带真实标签）已保存至：{out_path}")

def visualize_tsne_kmeans_with_indices(cfg, features, cluster_labels, indices=None, title="t-SNE of KMeans results"):
    """
    t-SNE 可视化 KMeans 结果，并标出指定 indices 样本
    参数:
        features: Tensor[N, D] 或 ndarray[N, D]，特征
        cluster_labels: Tensor[N] 或 ndarray[N]，KMeans 聚类结果标签
        indices: list/ndarray/Tensor，指定需要标注的样本索引
        title: 图标题
    """
    # 保证 numpy 格式
    if isinstance(features, torch.Tensor):
        features_np = features.detach().cpu().numpy()
    else:
        features_np = np.array(features)

    if isinstance(cluster_labels, torch.Tensor):
        cluster_labels = cluster_labels.detach().cpu().numpy()
    else:
        cluster_labels = np.array(cluster_labels)

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features_np)

    plt.figure(figsize=(8, 8))
    # 所有样本按聚类标签上色
    plt.scatter(features_2d[:, 0], features_2d[:, 1],
                c=cluster_labels, cmap="tab10", s=10, alpha=0.6, label="All samples")

    # 如果有 indices，单独标注
    if indices is not None and len(indices) > 0:
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        plt.scatter(features_2d[indices, 0], features_2d[indices, 1],
                    c="red", marker="x", s=40, label="Marked samples")

    plt.title(title)
    plt.legend()
    
    path = os.path.join(cfg['cdc_dir'], "visualization_kmeans_prun.png")
    plt.savefig(path)


def plot_tsne_moco(features, labels, title="t-SNE Visualization", class_names=None, save_path=None):
    
    """
    使用 t-SNE 对特征进行降维并可视化
    Args:
        features (Tensor or ndarray): 特征矩阵 [N, D]
        labels (Tensor or ndarray): 样本对应的类别标签 [N]
        title (str): 图标题
        class_names (list): 类别名称（可选）
        save_path (str): 若不为 None，则保存图片到该路径
    """
    
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    print(f"[t-SNE] Running on {features.shape[0]} samples with dim={features.shape[1]}")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    num_classes = len(np.unique(labels))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', s=8, alpha=0.7)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    if class_names is not None:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_names, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    if save_path is not None:
        save_path = os.path.join(save_path, f"tsne.png")
        plt.savefig(save_path, dpi=300)
        print(f"[t-SNE] Saved visualization to: {save_path}")
    #plt.show()

@torch.no_grad()
def calibration_evaluate(cfg: object, path: object, epoch: object, subhead_index: object, all_predictions: object,
                       title: object, class_names: object = None,
                       compute_confusion_matrix: object = True, confusion_matrix_file: object = None,
                         features: object = None, indices=None, flag=False, class_dis = False, remove: set = None,highconf: set = None, shake: set = None, bins: list=None) -> object:
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.
    # Hungarian matching

    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    outputs = head['outputs'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    reordered_probs = torch.zeros((num_elems, num_classes), dtype=probs.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
        reordered_probs[:, target_i] = probs[:, pred_i]
    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    #plot_tsne_moco(features, targets, save_path=cfg['cdc_dir'])

    remove_acc, highconf_acc, highconf_acc_balanced = None, None, None
    
    if remove is not None and len(remove) > 0:
        remove = torch.tensor(list(remove), device=targets.device, dtype=torch.long)
        # ---- remove 样本准确率 ----
        remove_preds = reordered_preds[remove]
        remove_targets = targets[remove]
        remove_acc = (remove_preds == remove_targets).float().mean().item()

        # ---- 高置信度样本准确率 (全局 top-k) ----
        sample_conf = probs.max(dim=1)[0]
        topk_conf, topk_idx = torch.topk(sample_conf, k=len(remove))
        highconf_preds = reordered_preds[topk_idx]
        highconf_targets = targets[topk_idx]
        highconf_acc = (highconf_preds == highconf_targets).float().mean().item()

        # ---- 类别均衡版高置信度准确率 ----
        k = len(remove)  # 目标总量
        per_class = max(1, k // num_classes)  # 每类选多少个
        selected_idx = []

        for cls_id in range(num_classes):
            cls_mask = (targets == cls_id)
            cls_conf = sample_conf.clone()
            cls_conf[~cls_mask] = -1  # 非该类设为无效
            # 取该类别前 per_class 个高置信度样本
            topk_cls_conf, topk_cls_idx = torch.topk(cls_conf, k=min(per_class, cls_mask.sum().item()))
            selected_idx.append(topk_cls_idx)

        if selected_idx:
            selected_idx = torch.cat(selected_idx)
            # 如果总量 < k，就从剩下的最高置信度里补足
            if len(selected_idx) < k:
                remain = torch.ones(num_elems, dtype=torch.bool, device=targets.device)
                remain[selected_idx] = False
                extra_conf = sample_conf.clone()
                extra_conf[~remain] = -1
                need = k - len(selected_idx)
                extra_conf_val, extra_conf_idx = torch.topk(extra_conf, k=need)
                selected_idx = torch.cat([selected_idx, extra_conf_idx])

            highconf_preds_bal = reordered_preds[selected_idx]
            highconf_targets_bal = targets[selected_idx]
            highconf_acc_balanced = (highconf_preds_bal == highconf_targets_bal).float().mean().item()

        print(f"[Epoch {epoch}] Remove size: {len(remove)}, "
              f"Remove acc: {remove_acc:.4f}, "
              f"High-conf acc: {highconf_acc:.4f}, num: {len(topk_idx)}, "
              f"High-conf balanced acc: {highconf_acc_balanced:.4f}, num: {len(selected_idx)}")
        
    
    shake_acc = None
    if shake is not None and len(shake) > 0:
        shake = torch.tensor(list(shake), device=targets.device, dtype=torch.long)

        # ---- remove 样本准确率 ----
        shake_preds = reordered_preds[shake]
        shake_targets = targets[shake]
        shake_acc = (shake_preds == shake_targets).float().mean().item()
        print(f"Shake acc: {shake_acc:.4f}, Shake size: {len(shake)} ")

        log_path = os.path.join(cfg['cdc_dir'], 'training_log.log')
        log_file = open(log_path, 'a')
        log_file.write(f"Shake acc: {shake_acc:.4f}, Shake size: {len(shake)}\n ")
        log_file.close()  

    high_acc = None
    if highconf is not None and len(highconf) > 0:
        highconf = torch.tensor(list(highconf), device=targets.device, dtype=torch.long)

        # ---- remove 样本准确率 ----
        shake_preds = reordered_preds[highconf]
        shake_targets = targets[highconf]
        high_acc = (shake_preds == shake_targets).float().mean().item()
        print(f"highconf acc: {high_acc:.4f}, highconf size: {len(highconf)} ")

        log_path = os.path.join(cfg['cdc_dir'], 'training_log.log')
        log_file = open(log_path, 'a')
        log_file.write(f"highconf acc: {high_acc:.4f}, Shake size: {len(highconf)}\n ")
        log_file.close()   
        #pdb.set_trace()



    if bins is not None:
        for i in range(len(bins)):
            #continue
            bin_ = torch.tensor(bins[i].tolist(), device=targets.device, dtype=torch.long)
            if len(bin_)>0:
                bin_preds = reordered_preds[bin_]
                bin_targets = targets[bin_]
                bin_acc = (bin_preds == bin_targets).float().mean().item()
                print(f"bin acc group{i}: {bin_acc:.4f}, bin size: {len(bin_)} ")
                log_path = os.path.join(cfg['cdc_dir'], 'training_log.log')
                log_file = open(log_path, 'a')
                log_file.write(f"bin accgroup{i}: {bin_acc:.4f}, bin size: {len(bin_)}\n ")
                log_file.close()   


            


    uniform_target = torch.full_like(outputs, 1.0 / num_classes).cuda()
    sample_losses = F.mse_loss(outputs, uniform_target, reduction='none')     
    sample_losses = sample_losses.mean(dim=1)
    hard_vector = []
    hard_vector.extend(sample_losses.tolist())
    
    #pdb.set_trace()

    probs = torch.softmax(outputs, dim=1)                   # 转为概率
    confidences = torch.max(probs, dim=1).values            # 每个样本的最大概率值
    logits = torch.max(outputs, dim=1).values   
    hard_vector = confidences.tolist()  
    #hard_vector = logits.tolist()         
    
    if indices is None:
        #print("111")
        #indices = get_indices_in_FlexRand_balanced(hard_vector, reordered_preds.cpu().numpy(), p_high=1.0, p_low=0.4, balance=True)
        
        indices = get_indices_in_range_balanced(hard_vector, reordered_preds.cpu().numpy(), p_high=1.0, p_low=0.3, balance=True)
        
        print(f"Indices : {len(indices), indices[:10]}")

        if flag:
            #pdb.set_trace()
            total_indices = set(range(num_elems))
            prun_indices = list(set(total_indices) - set(indices))
            visualize_tsne_kmeans_with_indices(
                cfg,
                features=features,
                cluster_labels=reordered_preds,
                indices=prun_indices,
                title=f"t-SNE epoch {epoch}"
            )
    
        print("Evaluating hard samples……")
        labels_hard = targets.cpu().numpy()[indices]
        pred_hard = reordered_preds.cpu().numpy()[indices]
        acc_h = np.sum(pred_hard == labels_hard) / len(labels_hard)
        print("hard acc", acc_h)

    if features is None:
        sc, chi, dbi = 0, 0, 0
    else:
        sc = metrics.silhouette_score(features.cpu().numpy(), predictions.cpu().numpy())
        chi = metrics.calinski_harabasz_score(features.cpu().numpy(), predictions.cpu().numpy())
        dbi = metrics.davies_bouldin_score(features.cpu().numpy(), predictions.cpu().numpy())

    #############
    # https://github.com/Impression2805/FMFP/blob/main/utils/metrics.py#L142
    AURC, EAURC = calc_aurc_eaurc(probs.cpu().tolist(), (reordered_preds == targets).cpu().tolist())
    AUROC, AUPR_success, AUPR_err, FPR95, TNR95 = calc_fpr_aupr(probs.cpu().tolist(),(reordered_preds == targets).cpu().tolist())
    ############

    #_, preds_top5 = probs.topk(5, 1, largest=True)
    _, preds_top5 = probs.topk(min(5,num_classes), 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)
    # ece = calibration_error(reordered_probs, targets, n_bins=15).item()
    conf = torch.stack([reordered_probs[idx,x] for idx,x in enumerate(reordered_preds)])
    ece = calculate_cdc_ece(conf.cpu(),reordered_preds.cpu(), targets.cpu())
    conf = conf.mean().item()
    # conf = reordered_probs.max(1)[0].mean().item()
    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(),
                         class_names, confusion_matrix_file)

    # cal calibration
    if (epoch == 0) or (epoch + 1) % cfg['cluster_eval']['plot_freq'] == 0:
        try:
            plot_acc_calibration(path, reordered_probs.cpu().numpy(),
                                 targets.cpu().numpy(), acc, ece,
                                 n_bins=15, title=title+'cali',
                                 epoch=epoch + 1)
        except:
            pass

    if class_dis:
        # ====== Cluster-wise evaluation (Hungarian matched) ======
        cluster_stats = {}
        for c in range(num_classes):
            # 找到簇 c 的样本索引
            idxs = (reordered_preds == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0:
                cluster_stats[c] = {
                    "num_samples": 0,
                    "mean_conf": 0.0,
                    "acc": 0.0
                }
                continue

            # 簇内样本数
            num_samples = idxs.size(0)

            # 簇内平均置信度
            mean_conf = confidences[idxs].mean().item()

            # 簇内准确率（因为 reordered_preds 已经做过匈牙利匹配）
            acc_c = (reordered_preds[idxs] == targets[idxs]).float().mean().item()

            cluster_stats[c] = {
                "num_samples": num_samples,
                "mean_conf": mean_conf,
                "acc": acc_c
            }

        # 打印结果
        print("\n=== Cluster-wise Stats (Hungarian matched) ===")
        for c, stats in cluster_stats.items():
            print(f"Cluster {c}: "
                f"Num={stats['num_samples']}, "
                f"MeanConf={stats['mean_conf']:.4f}, "
                f"Acc={stats['acc']:.4f}")

    wandb.log({
        'epoch_cali': epoch,
        'ACC_cali': round(acc*100, 6), 'NMI_cali': round(nmi*100, 6),
        'ARI_cali': round(ari*100, 6), 'ECE_cali': round(ece*100, 6), 'CONF_cali': round(conf*100, 6),

        'C_AURC': round(AURC * 100, 6), 'C_EAURC': round(EAURC * 100, 6), 'C_AUROC': round(AUROC * 100, 6),
        'C_AUPR_success': round(AUPR_success * 100, 6), 'C_AUPR_err': round(AUPR_err * 100, 6),
        'C_FPR95': round(FPR95 * 100, 6), 'C_TNR95': round(TNR95 * 100, 6),

        'C_SC': round(sc, 6), 'C_CHI': round(chi, 6), 'C_DBI': round(dbi, 6),

        'ACC Top-5_cali': round(top5*100, 6),
    })

    if flag:
        return {'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
            'ARI': round(ari*100, 6), 'ECE': round(ece*100, 6), 'CONF': round(conf*100, 6),
            'ACC Top-5': round(top5*100, 6),

            'AURC': round(AURC*100,6), 'EAURC': round(EAURC*100,6), 'AUROC': round(AUROC*100,6),
            'AUPR_success': round(AUPR_success*100,6), 'AUPR_err': round(AUPR_err*100,6),
            'FPR95': round(FPR95*100,6), 'TNR95': round(TNR95*100,6),

            'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6),

            'hungarian_match': match}, indices
    
    if remove is not None and len(remove) > 0:
        return {'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
            'ARI': round(ari*100, 6), 'ECE': round(ece*100, 6), 'CONF': round(conf*100, 6),
            'ACC Top-5': round(top5*100, 6),

            'AURC': round(AURC*100,6), 'EAURC': round(EAURC*100,6), 'AUROC': round(AUROC*100,6),
            'AUPR_success': round(AUPR_success*100,6), 'AUPR_err': round(AUPR_err*100,6),
            'FPR95': round(FPR95*100,6), 'TNR95': round(TNR95*100,6),

            'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6),

            'hungarian_match': match, "remove_acc": remove_acc, "highconf_acc": highconf_acc,

            "highconf_acc_balanced": highconf_acc_balanced}

    return {'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
            'ARI': round(ari*100, 6), 'ECE': round(ece*100, 6), 'CONF': round(conf*100, 6),
            'ACC Top-5': round(top5*100, 6),

            'AURC': round(AURC*100,6), 'EAURC': round(EAURC*100,6), 'AUROC': round(AUROC*100,6),
            'AUPR_success': round(AUPR_success*100,6), 'AUPR_err': round(AUPR_err*100,6),
            'FPR95': round(FPR95*100,6), 'TNR95': round(TNR95*100,6),

            'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6),

            'hungarian_match': match}





def hign_conf_evaluation(cfg, model, train_dataloader, clustering_stats):
    # Initialize conflabel_bins with -1
    conflabel_bins = torch.ones(len(train_dataloader.dataset), dtype=torch.int64).cuda()
    conflabel_bins *= -1
    
    # Initialize lists to store indices of correct and incorrect samples
    correct_indices = []
    incorrect_indices = []
    low_correct_indices = []
    low_incorrect_indices = []
    low_correct_targets = []
    low_incorrect_targets = []
    
    model.eval()

    total_acc, total_num = 0, 0
    
    high_conf_counter = Counter()
    low_conf_counter = Counter()
    neg_logits_per_class = defaultdict(list)
    pos_logits_per_class = defaultdict(list) 
    
    for i, batch in enumerate(train_dataloader):
        index = batch['index'].cuda()
        image = batch['image'].cuda()
        gt = batch['target'].cuda()
        gt_map = clustering_stats['hungarian_match']
        #print(gt_map)
        # Remap ground truth labels based on Hungarian matching
        for pre, post in gt_map:
            gt[batch['target'] == post] = pre
        
        with torch.no_grad():
            # Forward pass depending on method
            if cfg['method'] == 'cc':
                output = model(image.cuda(non_blocking=True), forward_pass='test')[0]
            elif cfg['method'] == 'tcl':
                logits, output = model.module.forward_all(image)
            else:
                output = model(image)[0]
                logits = output.clone()
        
        # Get predictions and probabilities
        if cfg['method'] == 'cc' or cfg['method'] == 'tcl':
            pred_prob, pred_label = torch.max(output, dim=1)
        else:
            pred_prob, pred_label = torch.max(F.softmax(output, dim=1), dim=1)
        
        # Create mask for high-confidence samples
        mask = (pred_prob > cfg['cluster_eval']['select_conf'])
        low_mask = (pred_prob < 0.6)
        
        # Calculate metrics for high-confidence samples
        total_num += mask.sum().item()
        correct_mask = (gt[mask] == pred_label[mask])
        correct_low_mask = (gt[low_mask] == pred_label[low_mask])
        total_acc += correct_mask.sum().item()
        
        # Update conflabel_bins
        conflabel_bins[index[mask]] = pred_label[mask]
        
        # Store indices of correct and incorrect samples
        correct_high_conf_indices = index[mask][correct_mask]
        incorrect_high_conf_indices = index[mask][~correct_mask]
        
        correct_low_conf_indices = index[low_mask][correct_low_mask]
        incorrect_low_conf_indices = index[low_mask][~correct_low_mask]
        
        correct_indices.extend(correct_high_conf_indices.cpu().numpy())
        incorrect_indices.extend(incorrect_high_conf_indices.cpu().numpy())
        
        low_correct_indices.extend(correct_low_conf_indices.cpu().numpy())
        low_incorrect_indices.extend(incorrect_low_conf_indices.cpu().numpy())
        
        low_correct_targets.extend(gt[low_mask][correct_low_mask])
        low_incorrect_targets.extend(gt[low_mask][~correct_low_mask])
        
        # 对高置信度样本的预测标签进行统计
        if mask.sum().item() > 0:
            # 这里使用torch.unique统计各类别的个数，也可以转换为numpy后用Counter或np.bincount
            unique_labels, counts = torch.unique(pred_label[mask], return_counts=True)
            batch_counter = Counter(dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy())))
            high_conf_counter += batch_counter
        
        # 对低置信度样本的预测标签进行统计
        if low_mask.sum().item() > 0:
            unique_labels, counts = torch.unique(pred_label[low_mask], return_counts=True)
            batch_counter = Counter(dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy())))
            low_conf_counter += batch_counter
            
        
            
        # negative outputs usage
        high_conf_logits = logits[mask]         # 选出高信 logits
        high_conf_preds = pred_label[mask]      # 高信预测类别

        """ for logit_vec, pred in zip(high_conf_logits, high_conf_preds):
            neg_logits = torch.cat([logit_vec[:pred], logit_vec[pred+1:]])  # 去掉预测类的 logits
            neg_logits_per_class[int(pred.item())].append(neg_logits.cpu().numpy()) """
            
        for logit_vec, pred in zip(high_conf_logits, high_conf_preds):
            pred = int(pred.item())              # 转为 Python int
            logit_vec = logit_vec.cpu()

            # 记录正输出（即预测类别的 logit）
            pos_logits_per_class[pred].append(logit_vec[pred].item())

            # 记录负输出（去掉预测类的 logit）
            neg_logits = torch.cat([logit_vec[:pred], logit_vec[pred+1:]])
            neg_logits_per_class[pred].append(neg_logits.numpy())
    
    gaussian_params = {}  # key = class_id, value = (mean, std)

    for class_id, neg_logits_list in neg_logits_per_class.items():
        # 将所有负 logits 拼接为一维
        flattened = np.concatenate(neg_logits_list)
        mu, std = norm.fit(flattened)
        gaussian_params[class_id] = (mu, std)
        print(f"Class {class_id} 的负输出高斯分布: μ = {mu:.4f}, σ = {std:.4f}")
        
    #import pdb; pdb.set_trace()    
    
    for class_id, pos_logits_list in pos_logits_per_class.items():
        #flattened = np.concatenate(pos_logits_list)
        mu, std = norm.fit(pos_logits_list)
        gaussian_params[class_id] = (mu, std)
        print(f"Class {class_id} 的正输出高斯分布: μ = {mu:.4f}, σ = {std:.4f}")
    
    # 输出统计结果
    print("高置信度样本类别分布：", high_conf_counter)
    print("低置信度样本类别分布：", low_conf_counter)
    
    # Print and log results
    try:
        print("select num:", total_acc,
              "total num:", total_num,
              "acc:", round(total_acc / total_num, 6))
        wandb.log({
            "select num": total_acc,
            "total num": total_num,
            "select acc": round(total_acc / total_num, 6)
        })
    except:
        pass
    
    print(correct_indices[:10])
    print(incorrect_indices[:10])
    
    print(low_correct_indices[:10])
    print(low_incorrect_indices[:10])
    
    # Return additional information
    return {'correct_indices': correct_indices, 'incorrect_indices': incorrect_indices, 'low_correct_indices': low_correct_indices, 'low_incorrect_indices': low_incorrect_indices}

def visualize_tsne_with_confidence(cfg, model, train_dataloader, features_extractor,clustering_stats, n_components=2, perplexity=30, n_iter=1000):
    """
    Create a t-SNE visualization that highlights high and low confidence samples,
    differentiating between correct and incorrect predictions.
    
    Args:
        cfg: Configuration dictionary
        model: Trained model
        train_dataloader: DataLoader for the training data
        features_extractor: Function to extract features from model
        n_components: Number of t-SNE components (default: 2)
        perplexity: t-SNE perplexity parameter (default: 30)
        n_iter: Number of iterations for t-SNE (default: 1000)
    
    Returns:
        None (displays and saves the plot)
    """
    from tqdm import tqdm
    
    print("Extracting features and confidence scores...")
    features = []
    labels = []
    pred_labels = []
    confidences = []
    indices = []
    
    model.eval()
    
    # Extract features and confidence scores
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            index = batch['index']
            image = batch['image'].cuda()
            target = batch['target'].cuda()
            gt_map = clustering_stats['hungarian_match']
            #print(gt_map)
            # Remap ground truth labels based on Hungarian matching
            for pre, post in gt_map:
                target[batch['target'] == post] = pre
            # Extract features
            if hasattr(features_extractor, '__call__'):
                batch_features = features_extractor(model, image)
                if cfg['method'] == 'scan':
                    output, _ = model(image, forward_pass='return_all')
                else:
                    output = model.module.forward_c(image)
            else:
                # Default feature extraction if no extractor is provided
                if cfg['method'] == 'cc':
                    output, batch_features = model(image, forward_pass='return_features')
                elif cfg['method'] == 'tcl':
                    output = model.module.forward_c(image)
                    batch_features = model.module.backbone(image)
                else:
                    output, batch_features = model(image, return_features=True)
                
                # Handle different feature shapes
                if isinstance(batch_features, tuple):
                    batch_features = batch_features[-1]  # Use the last feature layer
                
                if len(batch_features.shape) > 2:
                    batch_features = batch_features.mean([2, 3])  # Global average pooling for spatial features
            
            # Get predictions and confidences
            if cfg['method'] == 'cc' or cfg['method'] == 'tcl' or cfg['method'] == 'scan':
                pred_prob, pred = torch.max(output, dim=1)
            else:
                
                pred_prob, pred = torch.max(F.softmax(output, dim=1), dim=1)
            
            # Collect data
            features.append(batch_features.cpu().numpy())
            labels.append(target.cpu().numpy())
            pred_labels.append(pred.cpu().numpy())
            confidences.append(pred_prob.cpu().numpy())
            indices.append(index.numpy())
    
    # Concatenate all batches
    features = np.vstack(features)
    labels = np.concatenate(labels)
    pred_labels = np.concatenate(pred_labels)
    confidences = np.concatenate(confidences)
    indices = np.concatenate(indices)
    
    # Identify correct and incorrect predictions
    correct_mask = (labels == pred_labels)
    
    # Define high confidence threshold
    high_conf_threshold = cfg['cluster_eval']['select_conf']
    low_conf_threshold = 0.6
    
    # Create category masks
    high_conf_correct = (confidences > high_conf_threshold) & correct_mask
    high_conf_incorrect = (confidences > high_conf_threshold) & ~correct_mask
    low_conf_correct = (confidences < low_conf_threshold) & correct_mask
    low_conf_incorrect = (confidences < low_conf_threshold) & ~correct_mask
    mid_conf = ~(high_conf_correct | high_conf_incorrect | low_conf_correct | low_conf_incorrect)
    
    print(f"High confidence correct: {np.sum(high_conf_correct)}")
    print(f"High confidence incorrect: {np.sum(high_conf_incorrect)}")
    print(f"Low confidence correct: {np.sum(low_conf_correct)}")
    print(f"Low confidence incorrect: {np.sum(low_conf_incorrect)}")
    print(f"Medium confidence: {np.sum(mid_conf)}")
    
    # Perform t-SNE dimensionality reduction
    print(f"Running t-SNE with perplexity={perplexity}, n_iter={n_iter}...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, verbose=1, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Plot t-SNE results
    plt.figure(figsize=(12, 10))
    
    # Plot points with different colors based on confidence and correctness
    plt.scatter(features_tsne[mid_conf, 0], features_tsne[mid_conf, 1], 
                c='gray', alpha=0.3, s=10, label='Medium confidence')
    
    plt.scatter(features_tsne[low_conf_correct, 0], features_tsne[low_conf_correct, 1], 
                c='darkgreen', alpha=0.6, s=30, label='Low conf. correct')
    
    plt.scatter(features_tsne[low_conf_incorrect, 0], features_tsne[low_conf_incorrect, 1], 
                c='darkred', alpha=0.6, s=30, label='Low conf. incorrect')
    
    plt.scatter(features_tsne[high_conf_correct, 0], features_tsne[high_conf_correct, 1], 
                c='limegreen', alpha=0.8, s=50, marker='*', label='High conf. correct')
    
    plt.scatter(features_tsne[high_conf_incorrect, 0], features_tsne[high_conf_incorrect, 1], 
                c='red', alpha=0.8, s=50, marker='*', label='High conf. incorrect')
    
    # Add plot details
    plt.title(f't-SNE Visualization with Confidence Highlighting\nHigh Conf. Threshold: {high_conf_threshold}, Low Conf. Threshold: {low_conf_threshold}')
    plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), 
                 label='Ground Truth Class', alpha=0.3)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    
    # Save the figure
    save_dir = os.path.join(cfg.get('output_dir', './'), 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'tsne_confidence_viz.png'), dpi=300, bbox_inches='tight')
    
    print(f"Visualization saved to {save_dir}/tsne_confidence_viz.png")
    plt.show()

    # Optional: Save data for future use
    np.savez(os.path.join(save_dir, 'tsne_data.npz'),
             tsne_coords=features_tsne,
             labels=labels,
             pred_labels=pred_labels,
             confidences=confidences,
             indices=indices)
    
    return features_tsne, labels, pred_labels, confidences, indices

def create_feature_extractor(cfg, method_type):
    """
    Create a feature extractor function based on the method type
    
    Args:
        cfg: Configuration dictionary
        method_type: String indicating the model method type
        
    Returns:
        Feature extractor function
    """
    if method_type == 'cc':
        def extract_features(model, images):
            _, features = model(images, forward_pass='return_features')
            return features
        return extract_features
    
    elif method_type == 'tcl':
        def extract_features(model, images):
            features = model.module.backbone(images)
            if isinstance(features, tuple):
                features = features[-1]
            if len(features.shape) > 2:
                features = features.mean([2, 3])
            return features
        return extract_features
    
    elif method_type == 'scan':
        def extract_features(model, images):
            features = model.module.backbone(images)
            if isinstance(features, tuple):
                features = features[-1]
            if len(features.shape) > 2:
                features = features.mean([2, 3])
            return features
        return extract_features
    else:  # Default extractor
        def extract_features(model, images):
            _, features = model(images, return_features=True)
            if isinstance(features, tuple):
                features = features[-1]
            if len(features.shape) > 2:
                features = features.mean([2, 3])
            return features
        return extract_features

# hard example TSNE usage:
def run_tsne_visualization(cfg, model, train_dataloader, clustering_stats):
    """
    Run the full t-SNE visualization workflow, including high confidence evaluation
    
    Args:
        cfg: Configuration dictionary
        model: Trained model
        train_dataloader: DataLoader for the training data
        clustering_stats: Dictionary containing clustering statistics including Hungarian matching
    
    Returns:
        Confidence bins and t-SNE visualization
    """
    # First run the high confidence evaluation to get confidence bins
    conf_indices = hign_conf_evaluation(cfg, model, train_dataloader, clustering_stats)
    
    # Create a feature extractor based on the method
    feature_extractor = create_feature_extractor(cfg, cfg['method'])
    
    # Run t-SNE visualization
    tsne_results = visualize_tsne_with_confidence(
        cfg, 
        model, 
        train_dataloader, 
        feature_extractor,
        clustering_stats,
        n_components=2,
        perplexity=30,
        n_iter=1000
    )
    
    return conf_indices, tsne_results

@torch.no_grad()
def cdc_evaluate(cfg: object, path: object, epoch: object, subhead_index: object, all_predictions: object,
                       title: object, class_names: object = None,
                       compute_confusion_matrix: object = True, confusion_matrix_file: object = None) -> object:
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    reordered_probs = torch.zeros((num_elems, num_classes), dtype=probs.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
        reordered_probs[:, target_i] = probs[:, pred_i]
    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)
    ece = calibration_error(reordered_probs, targets, n_bins=15).item()

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(),
                         class_names, confusion_matrix_file)

    # cal calibration
    if (epoch == 0) or (epoch + 1) % cfg['cluster_eval']['plot_freq'] == 0:
        try:
            plot_acc_calibration(path, reordered_probs.cpu().numpy(),
                                 targets.cpu().numpy(), acc, ece,
                                 n_bins=15, title=title,
                                 epoch=epoch + 1)
        except:
            pass

    wandb.log({
        'epoch': epoch,
        'ACC': round(acc, 6), 'NMI': round(nmi, 6),
        'ARI': round(ari, 6), 'ACC Top-5': round(top5, 6),
        'ECE': round(ece, 6)
    })

    return {'ACC': round(acc, 6), 'NMI': round(nmi, 6),
            'ARI': round(ari, 6), 'ACC Top-5': round(top5, 6),
            'ECE': round(ece, 6), 'hungarian_match': match}

def save_cum_acc(cfg, predictions, clustering_stats, output_val, dir=None):
    rank_idx = output_val.max(1)[0].sort(descending=True)[1]
    data_num = len(predictions[0]['targets'])

    gt_map = clustering_stats['hungarian_match']
    reordered_preds = torch.zeros(data_num, dtype=predictions[0]['predictions'].dtype).cuda()
    for pre, post in gt_map:
        reordered_preds[predictions[0]['predictions'] == pre] = post
    step = int(data_num / 1000)
    import os
    if dir is None:
        f = open(os.path.join(cfg['test_dir'], cfg['name'] + "cum_acc.csv"), "w")
    else:
        f = open(os.path.join(dir, cfg['name'] + "cum_acc.csv"), "w")
    for topi in range(step, data_num + 1, step):
        cur_acc = (reordered_preds[rank_idx[:topi]] == predictions[0]['targets'][rank_idx[:topi]]).sum() / topi
        np.savetxt(f, np.column_stack((topi, cur_acc.item())), fmt='%d %.6f')
    f.close()
    print()

def kmeans_evaluate(cfg, predictions, features, random_state=10):
    from sklearn.cluster import KMeans
    clustermd = KMeans(n_clusters=cfg['backbone']['nclusters'],random_state=random_state)
    clustermd.fit(features.cpu().numpy())
    plabels = clustermd.labels_
    gt = predictions[0]['targets'].cpu().numpy()
    match = _hungarian_match(plabels, gt, preds_k=cfg['backbone']['nclusters'], targets_k=cfg['backbone']['nclusters'])

    reordered_preds = np.zeros(gt.shape[0])
    for pred_i, target_i in match:
        reordered_preds[plabels == int(pred_i)] = int(target_i)

    acc = int((reordered_preds == gt).sum()) / float(gt.shape[0])
    nmi = metrics.normalized_mutual_info_score(gt, plabels)
    ari = metrics.adjusted_rand_score(gt, plabels)

    sc, chi, dbi =0,0,0
    # sc = metrics.silhouette_score(features.cpu().numpy(), plabels)
    # chi = metrics.calinski_harabasz_score(features.cpu().numpy(), plabels)
    # dbi = metrics.davies_bouldin_score(features.cpu().numpy(), plabels)

    wandb.log({
        'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
        'ARI': round(ari*100, 6),
        'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6),
    })
    print({
        'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
        'ARI': round(ari*100, 6), 'MATCH': match,
        'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6)
    })

def py_kmeans_evaluate(cfg, predictions, features, random_state=10):
    from cdc.utils.torch_clustering import PyTorchKMeans
    clustermd = PyTorchKMeans(n_clusters=cfg['backbone']['nclusters'],random_state=random_state)
    plabels = clustermd.fit_predict(features)
    plabels = plabels.cpu().numpy()
    gt = predictions[0]['targets'].cpu().numpy()
    match = _hungarian_match(plabels, gt, preds_k=cfg['backbone']['nclusters'], targets_k=cfg['backbone']['nclusters'])

    reordered_preds = np.zeros(gt.shape[0])
    for pred_i, target_i in match:
        reordered_preds[plabels == int(pred_i)] = int(target_i)

    acc = int((reordered_preds == gt).sum()) / float(gt.shape[0])
    nmi = metrics.normalized_mutual_info_score(gt, plabels)
    ari = metrics.adjusted_rand_score(gt, plabels)

    sc, chi, dbi =0,0,0
    # sc = metrics.silhouette_score(features.cpu().numpy(), plabels)
    # chi = metrics.calinski_harabasz_score(features.cpu().numpy(), plabels)
    # dbi = metrics.davies_bouldin_score(features.cpu().numpy(), plabels)

    wandb.log({
        'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
        'ARI': round(ari*100, 6),
        'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6),
    })
    print({
        'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
        'ARI': round(ari*100, 6), 'MATCH': match,
        'SC': round(sc, 6), 'CHI': round(chi, 6), 'DBI': round(dbi, 6)
    })

def plot_tsne(cfg, model, train_dataloader):
    z_features_numpy = np.zeros((len(train_dataloader.sampler), cfg['backbone']['feat_dim']))
    proj_features_numpy = np.zeros((len(train_dataloader.sampler), cfg['backbone']['proj_output_dim']))
    gt = np.zeros((len(train_dataloader.sampler)), dtype='int')
    pl_numpy = np.zeros((len(train_dataloader.sampler)), dtype='int')

    ptr = 0
    model.eval()
    with torch.no_grad():
        for batch in train_dataloader:
            images = batch['image'].cuda(non_blocking=True)
            bs = images.shape[0]
            res = model(images, forward_pass='return_all')
            z_features_numpy[ptr: ptr + bs] = res['features'].cpu().numpy()
            proj_features_numpy[ptr: ptr + bs] = res['projection'].cpu().numpy()
            gt[ptr: ptr + bs] = batch['target'].cpu().numpy()
            pl_numpy[ptr: ptr + bs] = res['output'][0].cpu().numpy().max(1)[1]
            ptr += bs

    z_center_numpy = np.zeros((cfg['backbone']['nclusters'], cfg['backbone']['feat_dim']))
    for label_idx in range(cfg['backbone']['nclusters']):
        center_ = z_features_numpy[gt==label_idx].mean(0)
        z_ = z_features_numpy[gt==label_idx]
        similarity = 0
        for sample in z_:
            similarity_ = torch.cosine_similarity(torch.from_numpy(sample),torch.from_numpy(center_),dim=0)
            if similarity_ > similarity:
                similarity = similarity_
                z_center_numpy[label_idx] = sample
    z_features_wrap = np.concatenate((z_features_numpy,z_center_numpy))

    # tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(z_features_numpy)
    tsne_results = tsne.fit_transform(proj_features_numpy)
    # df_subset['tsne-2d-one'] = tsne_results[:, 0]
    # df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset = pd.DataFrame()
    # df_subset["y"] = pl_numpy
    df_subset["y"] = gt
    df_subset["comp-1"] = tsne_results[:5000, 0]
    df_subset["comp-2"] = tsne_results[:5000, 1]

    center_subset= pd.DataFrame()
    center_subset["comp-1"] = tsne_results[5000:, 0]
    center_subset["comp-2"] = tsne_results[5000:, 1]

    plt.figure(figsize=(6, 6), dpi=300)
    sns.scatterplot(
        x="comp-1", y="comp-2",
        hue="y",
        palette=sns.color_palette("hls", cfg['backbone']['nclusters']),
        data=df_subset,
        legend="full",
        # alpha=0.3
    )
    sns.scatterplot(
        x="comp-1", y="comp-2",
        # hue="y",
        # palette=sns.color_palette("hls", cfg['backbone']['nclusters']),
        data=center_subset,
        legend="full",
        # alpha=0.3
    )
    plt.show()
    print()

def plot_tsne_perepoch(cfg, proj_features_numpy, pred_numpy, gt_numpy, center_numpy):

    # tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(z_features_numpy)
    tsne_results = tsne.fit_transform(proj_features_numpy)
    # df_subset['tsne-2d-one'] = tsne_results[:, 0]
    # df_subset['tsne-2d-two'] = tsne_results[:, 1]
    df_subset = pd.DataFrame()
    df_subset["y"] = gt_numpy
    # df_subset["y"] = gt
    df_subset["comp-1"] = tsne_results[:, 0]
    df_subset["comp-2"] = tsne_results[:, 1]

    plt.figure(figsize=(6, 6), dpi=300)
    sns.scatterplot(
        x="comp-1", y="comp-2",
        hue="y",
        palette=sns.color_palette("hls", cfg['backbone']['nclusters']),
        data=df_subset,
        legend="full",
        # alpha=0.3
    )
    plt.show()

#######################AURC FRP95########################
# Calc coverage, risk
def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list
# Calc aurc, eaurc
def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    # print("AURC {0:.2f}".format(aurc*1000))
    # print("EAURC {0:.2f}".format(eaurc*1000))

    return aurc, eaurc
# AURC, EAURC
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    auroc = metrics.auc(fpr, tpr)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]
    tnr_in_tpr_95 = 1 - fpr[np.argmax(tpr >= .95)]

    precision, recall, thresholds = metrics.precision_recall_curve(correctness, softmax_max)
    aupr_success = metrics.auc(recall, precision)
    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)

    # print("AUROC {0:.2f}".format(auroc * 100))
    # print('AUPR_Success {0:.2f}'.format(aupr_success * 100))
    # print("AUPR_Error {0:.2f}".format(aupr_err*100))
    # print('FPR@TPR95 {0:.2f}'.format(fpr_in_tpr_95*100))
    # print('TNR@TPR95 {0:.2f}'.format(tnr_in_tpr_95 * 100))

    return auroc, aupr_success, aupr_err, fpr_in_tpr_95, tnr_in_tpr_95

def calculate_ece(probs, preds, targets, num_bins=15):
    """
    Calculate the Expected Calibration Error (ECE).

    Parameters:
    probs (torch.Tensor): Tensor of predicted probabilities.
    preds (torch.Tensor): Tensor of predicted labels.
    targets (torch.Tensor): Tensor of true labels.
    num_bins (int): Number of bins to use for calibration.

    Returns:
    float: Expected Calibration Error (ECE).
    """
    # 获取最大概率和预测标签
    confidences, predicted_classes = torch.max(probs, dim=1)  # confidences: [N], predicted_classes: [N]
    print("conf avg:", confidences.mean())
    predicted_classes = preds
    #print(probs[0], confidences[:10], predicted_classes[:10],targets[:10])
    # 创建 bin 的边界
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.tensor(0.0, device=probs.device)  # 确保 ece 使用正确的设备
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 选择当前 bin 中的样本
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)  # 在当前 bin 范围内的样本
        num_in_bin = in_bin.sum().item()  # 当前 bin 中样本的数量
        #num_in_bin = in_bin.float.mean()
        if num_in_bin > 0:
            # 计算当前 bin 的准确率和平均置信度
            accuracy_in_bin = (predicted_classes[in_bin] == targets[in_bin]).float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            # 计算 ECE 的贡献
            prop_in_bin = num_in_bin / probs.size(0)  # 当前 bin 中样本占总样本的比例
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

def calculate_cdc_ece(probs, preds, targets, num_bins=15):
    """
    Calculate the Expected Calibration Error (ECE).

    Parameters:
    probs (torch.Tensor): Tensor of predicted probabilities.
    preds (torch.Tensor): Tensor of predicted labels.
    targets (torch.Tensor): Tensor of true labels.
    num_bins (int): Number of bins to use for calibration.

    Returns:
    float: Expected Calibration Error (ECE).
    """
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.tensor(0.0)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find the indices of the predictions that fall into this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            # Compute accuracy and confidence for this bin
            accuracy_in_bin = (preds[in_bin] == targets[in_bin]).float().mean()
            avg_confidence_in_bin = probs[in_bin].mean()

            # Contribution to ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()

# TCL
def reorder_preds(predictions, targets):
    predictions = torch.from_numpy(predictions).cuda()
    targets = torch.from_numpy(targets).cuda()
    class_num = len(np.unique(targets.cpu()))
    match = _hungarian_match(predictions, targets, preds_k=class_num, targets_k=class_num)
    reordered_preds = torch.zeros(predictions.shape[0], dtype=predictions.dtype).cuda()
    #pdb.set_trace()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
    return reordered_preds.cpu().numpy()

def get_reverse_match(predictions, targets, osr=False):
    if osr:
        unique_classes = np.unique(np.concatenate((predictions, targets)))
        num_classes = len(unique_classes)
        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
        cost_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        for t, p in zip(targets, predictions):
            cost_matrix[class_to_index[t], class_to_index[p]] += 1  
        cost_matrix = -cost_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        match = {unique_classes[r]: unique_classes[c] for r, c in zip(row_ind, col_ind)}
        
        targets = torch.from_numpy(targets).cuda()
        predictions = torch.from_numpy(predictions).cuda()
        
    else:
        predictions = torch.from_numpy(predictions).cuda()
        targets = torch.from_numpy(targets).cuda()
        pre_num = len(np.unique(predictions.cpu()))
        class_num = len(np.unique(targets.cpu()))
        match = _hungarian_match(targets, predictions, targets_k=class_num, preds_k=class_num)
        match = {pred_i: target_i for pred_i, target_i in match}

    #pdb.set_trace()

    # 创建一个空的张量来存储映射后的结果
    mapped_labels = torch.zeros_like(targets).cuda()
    for i in range(targets.shape[0]):
        mapped_labels[i] = match[targets[i].item()]
    
    # 反向映射字典
    reverse_match = {v: k for k, v in match.items()}
    
    # 给定的 reordered_preds
    reordered_preds = predictions

    # 创建一个空的张量来存储反向映射后的结果
    original_preds = torch.zeros_like(reordered_preds).cuda()

    # 进行反向映射
    for i in range(reordered_preds.shape[0]):
        original_preds[i] = reverse_match[reordered_preds[i].item()]

    # 输出反向映射的结果
    return original_preds.cpu().numpy(), mapped_labels.cpu().numpy()

def cluster_metric(label, pred, osr=False):
    # The above code is calculating the normalized mutual information score (NMI) and adjusted Rand
    # index (ARI) between two sets of labels. The `metrics.normalized_mutual_info_score` function
    # computes the NMI between the true labels (`label`) and predicted labels (`pred`), while the
    # `metrics.adjusted_rand_score` function calculates the ARI between the same sets of labels. These
    # metrics are commonly used in clustering and classification tasks to evaluate the similarity
    # between two label assignments.
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    # pred_adjusted = get_y_preds(label, pred, len(set(label)))
    
    if osr:
        unique_classes = np.unique(np.concatenate((pred, label)))
        num_classes = len(unique_classes)

        class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

        cost_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        for t, p in zip(label, pred):
            cost_matrix[class_to_index[t], class_to_index[p]] += 1  
        cost_matrix = -cost_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        match = {unique_classes[c]: unique_classes[r] for r, c in zip(row_ind, col_ind)}
        #pdb.set_trace()
        pred_adjusted =  np.array([match[p] for p in pred])
        
    else:
        pred_adjusted = reorder_preds(pred, label)
        
    #pdb.set_trace()
        
    acc_all = metrics.accuracy_score(pred_adjusted, label)
    
    # 找出所有类别
    classes = np.unique(label)
    acc_per_class = {}

    # 针对每个类别计算准确率
    for cls in classes:
        # 筛选出该类别的样本索引
        idx = np.where(label == cls)[0]
        # 若某个类别没有样本，则跳过该类别
        if idx.size == 0:
            continue
        # 计算该类别上的准确率
        acc = metrics.accuracy_score(label[idx], pred_adjusted[idx])
        acc_per_class[cls] = acc

    # 输出每个类别的准确率
    for cls, acc in acc_per_class.items():
        print(f"类别 {cls} 的准确率: {acc:.4f}")
    
    return nmi * 100, ari * 100, acc_all * 100

def get_gauss(path, data_loader, model, osr=False, known=None):
    
    model.eval()
    pred_vector = []
    labels_vector = []
    logits_vector = []
    if osr:
        class_conf = {i: [] for i in range(len(known))}

    # Iterate through the data loader
    for data in data_loader:
        images = data['image']
        labels = data['target']
        # Move images and labels to CUDA device
        images = images.cuda(non_blocking=True)

        # Compute output
        with torch.amp.autocast('cuda'):
            preds = model.module.forward_c(images)
            preds = torch.argmax(preds, dim=1)
            logits = model.module.forward_osr(images)
            logits, _ = torch.max(logits, dim=1)  
            
        # Collect predictions and labels
        pred_vector.extend(preds.cpu().detach().numpy())
        labels_vector.extend(labels)
        logits_vector.extend(logits.cpu().detach().numpy())
    
    # Convert to numpy arrays for metric computation
    pred_vector = np.array(pred_vector)
    labels_vector = np.array(labels_vector)
    
    if osr:
        #mapping = {i: k for i, k in enumerate(known)}
        #pred_vector = np.array([mapping[p] for p in pred_vector])

        print("Pred shape {}, Label shape {}".format(pred_vector.shape, labels_vector.shape))

        
        #r_mapping = {k: i for i, k in enumerate(known)}

        #ece_preds = np.array([r_mapping[p] for p in pred_vector])

        print("train_gauss: ")

        #pdb.set_trace()
        filtered_logits = np.array(logits_vector)
        for i in range(len(pred_vector)):
            class_conf[pred_vector[i]].append(filtered_logits[i])
        class_gaussian_params = {}
        for class_id, conf_values in class_conf.items():
            if len(conf_values) > 0:  # 确保有数据
                mean, std = norm.fit(conf_values)
                class_gaussian_params[class_id] = {'mean': mean, 'std': std}
                print(f"Class {class_id}: Mean = {mean:.4f}, Std = {std:.4f}, Num = {len(conf_values)}")
            else:
                print(f"Class {class_id}: No data available")

    return class_gaussian_params

def compute_score(value, class_stats):
    mean = class_stats['mean']
    std = class_stats['std']
    
    # 确保标准差不为零
    if std == 0:
        return 1.0 if value >= mean else 0.0
    
    # 计算累积分布函数（CDF）作为得分
    score = norm.cdf(value, loc=mean, scale=std)
    
    return score

@torch.no_grad()
def evaluate_tcl(path, data_loader, model, osr=False, known=None, class_gaussian_params=None):
    #metric_logger = misc.MetricLogger(delimiter="  ")
    # Switch to evaluation mode
    model.eval()
    pred_vector = []
    labels_vector = []
    con_vector = []
    logits_vector = []
    soft_vector= []
    score_list = []

    if osr:
        class_conf = {i: [] for i in range(len(known))}

    # Iterate through the data loader
    for data in data_loader:
        images = data[0]
        labels = data[1]
        # Move images and labels to CUDA device
        images = images.cuda(non_blocking=True)

        # Compute output
        with torch.amp.autocast('cuda'):
            preds = model.module.forward_c(images)
            conf = preds
            preds = torch.argmax(preds, dim=1)
            soft, _ = torch.max(conf, dim=1)  
            logits = model.module.forward_osr(images)
            logits, _ = torch.max(logits, dim=1)  
            
        # Collect predictions and labels
        pred_vector.extend(preds.cpu().detach().numpy())
        labels_vector.extend(labels.numpy())
        con_vector.extend(conf.cpu().detach().numpy())
        soft_vector.extend(soft.cpu().detach().numpy())
        logits_vector.extend(logits.cpu().detach().numpy())
    
    # Convert to numpy arrays for metric computation
    pred_vector = np.array(pred_vector)
    labels_vector = np.array(labels_vector)
    
    if osr:
        mapping = {i: k for i, k in enumerate(known)}
        pred_clone = pred_vector
        pred_vector = np.array([mapping[p] for p in pred_vector])

        print("Pred shape {}, Label shape {}".format(pred_vector.shape, labels_vector.shape))

        filtered_indices = np.isin(labels_vector, known)
        filtered_list = filtered_indices.astype(int).tolist()

        #pdb.set_trace()
        filtered_preds = pred_vector[filtered_indices]
        filtered_labels = labels_vector[filtered_indices]
        con_vector = np.array(con_vector)
        #pdb.set_trace()
        
        reordered_preds, label_ordered = get_reverse_match(filtered_preds, filtered_labels, osr)
        
        filtered_confs = con_vector[filtered_indices]  # 置信度值（用于 ECE 计算）
        
        r_mapping = {k: i for i, k in enumerate(known)}

        ece_preds = np.array([r_mapping[p] for p in filtered_preds])
        ece_labels = np.array([r_mapping[p] for p in label_ordered])
        
        ece = calculate_ece(torch.tensor(filtered_confs).cpu(), 
                            torch.tensor(ece_preds).cpu(), 
                            torch.tensor(ece_labels).cpu())
        
        nmi, ari, acc = cluster_metric(filtered_labels, filtered_preds, osr)
        print(acc)
        
        filtered_logits = np.array(logits_vector)[filtered_indices]
        for i in range(len(ece_preds)):
            class_conf[ece_preds[i]].append(filtered_logits[i])
        
        auroc_score=None
        if class_gaussian_params:   
            for i in range(len(pred_clone)):
                score = compute_score(logits_vector[i], class_gaussian_params[pred_clone[i]])
                score_list.append(score)
            
            auroc_score = roc_auc_score(filtered_list, score_list)
        
        auroc_logits = roc_auc_score(filtered_list, logits_vector)
        auroc_soft = roc_auc_score(filtered_list, soft_vector)
        
        print(nmi, ari, acc, ece)
        print('AUROC: ', auroc_logits, auroc_soft, auroc_score)
        #pdb.set_trace()
        plot_acc_calibration(path, torch.tensor(filtered_confs).cpu().numpy(),
                             ece_labels, acc/100.0, ece,
                             n_bins = 15, title= 'calibration',
                             epoch=999)
        #pdb.set_trace()
            
        class_gaussian_params = {}
        for class_id, conf_values in class_conf.items():
            if len(conf_values) > 0:  # 确保有数据
                mean, std = norm.fit(conf_values)
                class_gaussian_params[class_id] = {'mean': mean, 'std': std}
                print(f"Class {class_id}: Mean = {mean:.4f}, Std = {std:.4f}, Num = {len(conf_values)}")
            else:
                print(f"Class {class_id}: No data available")
    else:
        
        #import pdb; pdb.set_trace()
        
        reordered_preds, label_ordered = get_reverse_match(pred_vector, labels_vector)
        ece = calculate_ece(torch.tensor(con_vector).cpu(), torch.tensor(reordered_preds).cpu(), torch.tensor(labels_vector).cpu())
        
        nmi, ari, acc = cluster_metric(labels_vector, pred_vector)
        
        print(nmi, ari, acc, ece)
        plot_acc_calibration(path, torch.tensor(con_vector).cpu().numpy(),
                                label_ordered, acc/100.0, ece,
                                n_bins = 15, title= 'calibration',
                                epoch=999)
        
    targets = torch.from_numpy(labels_vector).cuda()
    class_num = len(np.unique(targets.cpu()))
    match = _hungarian_match(pred_vector, labels_vector, preds_k=class_num, targets_k=class_num)
    
    
    return {'ACC': round(acc,6),'NMI': round(nmi,6),
            'ARI': round(ari,6), 'hungarian_match': match}
    
def evaluate_tcl_mask(path, data_loader, model, osr=False, known=None, class_gaussian_params=None, t=0.7 , method="tcl", conf_indices=None):
    #metric_logger = misc.MetricLogger(delimiter="  ")
    # Switch to evaluation mode
    model.eval()
    pred_vector = []
    labels_vector = []
    con_vector = []
    logits_vector = []
    all_logits_vector = []
    soft_vector= []
    score_list = []
    
    p_logits_vector = []
    p_pred_vector = []
    
    index_list = []
            
    # 指定你希望hook的卷积层
    final_conv_layer = "module.backbone.layer4"
    feature_maps = None
    
    # 定义hook函数，用于捕获指定层的输出特征图
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output

    # 注册hook
    for name, module in model.named_modules():
        if name == final_conv_layer:
            module.register_forward_hook(hook_fn)
    
    if osr:
        class_conf = {i: [] for i in range(len(known))}

    # Iterate through the data loader
    for data in data_loader:
        images = data[0]
        labels = data[1]
        index = data[2]
        
        index_list.extend(index.numpy())
        # Move images and labels to CUDA device
        images = images.cuda(non_blocking=True)

        # Compute output
        with torch.amp.autocast('cuda'):
            if method=="su":
                logits = model(images.cuda(non_blocking=True),
                        forward_pass='return_all')['output'][0]
                preds = F.softmax(logits, dim=1)
            else:
                logits, preds = model.module.forward_all(images)
                
            preds = torch.argmax(preds, dim=1)
            logits, _ = torch.max(logits, dim=1)  
        
        p_pred_vector.extend(preds.cpu().detach().numpy())
        p_logits_vector.extend(logits.cpu().detach().numpy())   
            
        cam = torch.mean(feature_maps, dim=1, keepdim=True)  # [B, 1, H, W]
        
        #import pdb; pdb.set_trace()
        
        # 对CAM归一化到 [0, 1]
        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cam_min  = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam_max  = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        # 设置阈值，将 CAM 中高响应部分遮挡掉
        # 这里将大于阈值的区域设为 0；其它区域保留原始信息（乘以1）
        threshold = t  # 根据需要调整
        mask = (cam_norm < threshold).float()   # 高响应区域 mask 值为0，低响应区域为1
        
        # 若 CAM 尺寸与输入图片尺寸不同，则进行上采样
        if mask.shape[2:] != images.shape[2:]:
            mask = torch.nn.functional.interpolate(mask, size=images.shape[2:], mode='bilinear', align_corners=False)
        
        # 对图片应用mask：遮住高CAM响应区域
        images_masked = images * mask
        
        # 使用遮挡后的图片再次做前向传播得到最终预测
        with torch.amp.autocast('cuda'):
            if method=="su":
                logits = model(images_masked.cuda(non_blocking=True),
                        forward_pass='return_all')['output'][0]
                preds = F.softmax(logits, dim=1)
            else:
                all_logits, preds = model.module.forward_all(images_masked)
            conf = preds
            preds = torch.argmax(preds, dim=1)
            soft, _ = torch.max(conf, dim=1)  
            logits, _ = torch.max(all_logits, dim=1)  
            
        # Collect predictions and labels
        pred_vector.extend(preds.cpu().detach().numpy())
        labels_vector.extend(labels.numpy())
        con_vector.extend(conf.cpu().detach().numpy())
        soft_vector.extend(soft.cpu().detach().numpy())
        logits_vector.extend(logits.cpu().detach().numpy())
        all_logits_vector.extend(all_logits.cpu().detach().numpy())
        
    # Convert to numpy arrays for metric computation
    pred_vector = np.array(pred_vector)
    labels_vector = np.array(labels_vector)
    index_vector = np.array(index_list)
    
    if osr:
        mapping = {i: k for i, k in enumerate(known)}
        pred_clone = pred_vector
        pred_vector = np.array([mapping[p] for p in pred_vector])

        print("Pred shape {}, Label shape {}".format(pred_vector.shape, labels_vector.shape))

        filtered_indices = np.isin(labels_vector, known)
        filtered_list = filtered_indices.astype(int).tolist()

        #pdb.set_trace()
        filtered_preds = pred_vector[filtered_indices]
        filtered_labels = labels_vector[filtered_indices]
        con_vector = np.array(con_vector)
        #pdb.set_trace()
        
        reordered_preds, label_ordered = get_reverse_match(filtered_preds, filtered_labels, osr)
        
        filtered_confs = con_vector[filtered_indices]  # 置信度值（用于 ECE 计算）
        
        r_mapping = {k: i for i, k in enumerate(known)}

        ece_preds = np.array([r_mapping[p] for p in filtered_preds])
        ece_labels = np.array([r_mapping[p] for p in label_ordered])
        
        ece = calculate_ece(torch.tensor(filtered_confs).cpu(), 
                            torch.tensor(ece_preds).cpu(), 
                            torch.tensor(ece_labels).cpu())
        
        nmi, ari, acc = cluster_metric(filtered_labels, filtered_preds, osr)
        print(acc)
        
        filtered_logits = np.array(logits_vector)[filtered_indices]
        for i in range(len(ece_preds)):
            class_conf[ece_preds[i]].append(filtered_logits[i])
        
        auroc_score=None
        if class_gaussian_params:   
            for i in range(len(pred_clone)):
                score = compute_score(logits_vector[i], class_gaussian_params[pred_clone[i]])
                score_list.append(score)
            
            auroc_score = roc_auc_score(filtered_list, score_list)
        
        auroc_logits = roc_auc_score(filtered_list, logits_vector)
        auroc_soft = roc_auc_score(filtered_list, soft_vector)
        
        print(nmi, ari, acc, ece)
        print('AUROC: ', auroc_logits, auroc_soft, auroc_score)
        #pdb.set_trace()
        plot_acc_calibration(path, torch.tensor(filtered_confs).cpu().numpy(),
                             ece_labels, acc/100.0, ece,
                             n_bins = 15, title= 'calibration',
                             epoch=999)
        #pdb.set_trace()
            
        class_gaussian_params = {}
        for class_id, conf_values in class_conf.items():
            if len(conf_values) > 0:  # 确保有数据
                mean, std = norm.fit(conf_values)
                class_gaussian_params[class_id] = {'mean': mean, 'std': std}
                print(f"Class {class_id}: Mean = {mean:.4f}, Std = {std:.4f}, Num = {len(conf_values)}")
            else:
                print(f"Class {class_id}: No data available")
    else:
        reordered_preds, label_ordered = get_reverse_match(pred_vector, labels_vector)
        ece = calculate_ece(torch.tensor(con_vector).cpu(), torch.tensor(reordered_preds).cpu(), torch.tensor(labels_vector).cpu())
        
        nmi, ari, acc = cluster_metric(labels_vector, pred_vector)
        
        print(nmi, ari, acc, ece)
        plot_acc_calibration(path, torch.tensor(con_vector).cpu().numpy(),
                                label_ordered, acc/100.0, ece,
                                n_bins = 15, title= 'calibration',
                                epoch=999)
        
    targets = torch.from_numpy(labels_vector).cuda()
    class_num = len(np.unique(targets.cpu()))
    match = _hungarian_match(pred_vector, labels_vector, preds_k=class_num, targets_k=class_num)
    
    # ratio
    correct_indices = conf_indices['correct_indices']
    correct_mismatch_count = sum(
        p_pred_vector[i] != pred_vector[i] for i in correct_indices
    )
    correct_mismatch_ratio = correct_mismatch_count / len(correct_indices)
    incorrect_indices = conf_indices['incorrect_indices']
    incorrect_mismatch_count = sum(
        p_pred_vector[i] != pred_vector[i] for i in incorrect_indices
    )
    incorrect_mismatch_ratio = incorrect_mismatch_count / len(incorrect_indices)
    print(f"Correct prediction mismatch ratio: {correct_mismatch_ratio:.4f}")
    print(f"Incorrect prediction mismatch ratio: {incorrect_mismatch_ratio:.4f}")
    
    #consistance_mask = (p_pred_vector == pred_vector)
    
    consistance_acc = (len(correct_indices)-correct_mismatch_count)/(len(incorrect_indices)+len(correct_indices)-incorrect_mismatch_count-correct_mismatch_count)
    print(f"Consistance accuracy: {consistance_acc:.4f}")
    
    # 提取 logits 的 float 值
    p_confidences = [x for x in p_logits_vector]
    mask_confidences = [x for x in logits_vector]

    # 计算 logits 变化值（原图 - mask图）
    logits_change = [p - m for p, m in zip(p_confidences, mask_confidences)]
    
    correct_indices = conf_indices['correct_indices']
    incorrect_indices = conf_indices['incorrect_indices']
    high_indices = correct_indices + incorrect_indices

    #negative
    negative_outputs(all_logits_vector, index_vector, high_indices)

    # 将 correct_indices 转为 set 加速查找
    correct_index_set = set(correct_indices)
    incorrect_index_set = set(incorrect_indices)
    # 找出 index_vector 中在 correct_indices 里的 logits_change 值
    correct_logits_change = [
        logits_change[i] for i in range(len(index_vector))
        if index_vector[i] in correct_index_set
    ]
    incorrect_logits_change = [
        logits_change[i] for i in range(len(index_vector))
        if index_vector[i] in incorrect_index_set
    ]

    plt.figure(figsize=(10, 5))

    # 横轴为排序后的索引
    plt.scatter(range(len(correct_logits_change)), correct_logits_change, 
                label='Correct Predictions', color='green', alpha=0.6, s=20)

    plt.scatter(range(len(incorrect_logits_change)), incorrect_logits_change, 
                label='Incorrect Predictions', color='red', alpha=0.6, s=20)

    plt.title("Sorted Logits Change After Masking")
    plt.xlabel("Sorted Sample Index")
    plt.ylabel("Logits Change (Original - Masked)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(path, 'logits_change.png'), dpi=300)
    
    #import pdb; pdb.set_trace()
    
    return {'ACC': round(acc,6),'NMI': round(nmi,6),
            'ARI': round(ari,6), 'hungarian_match': match}    

def evaluate_hard(path, data_loader, model, num_clusters, _indices=None, tsne = False):
    #metric_logger = misc.MetricLogger(delimiter="  ")
    # Switch to evaluation mode
    model.eval()
    pred_vector = []
    labels_vector = []
    con_vector = []
    hard_vector = []
    features_vector = []
    #index_list = []
    # Iterate through the data loader
    for data in data_loader:
        images = data[0]
        labels = data[1]
        #index = data[2]
        
        #index_list.extend(index.numpy())
        """ images = data['image']
        labels = data['target'] """
        # Move images and labels to CUDA device
        images = images.cuda(non_blocking=True)

        # Compute output
        with torch.amp.autocast('cuda'):
            #import pdb; pdb.set_trace()
            
            features, preds = model.module.forward_zc(images)  # 假设模型有一个backbone方法来提取特征
            features_vector.extend(features.cpu().detach().numpy())
            
            """ uniform_target = torch.full_like(preds, 1.0 / num_clusters).cuda()
            sample_losses = F.mse_loss(preds, uniform_target, reduction='none')
            
            sample_losses = sample_losses.mean(dim=1) """
            
            #sample_losses = preds.max(dim=1)[0]
            
            #logits
            preds = model.module.forward_osr(images)
            sample_losses = preds.max(dim=1)[0]
        
            conf = preds
            preds = torch.argmax(preds, dim=1) 
    
        # Collect predictions and labels
        hard_vector.extend(sample_losses.tolist())
        pred_vector.extend(preds.cpu().detach().numpy())
        labels_vector.extend(labels.numpy())
        con_vector.extend(conf.cpu().detach().numpy())
    
    # Convert to numpy arrays for metric computation
    pred_vector = np.array(pred_vector)
    labels_vector = np.array(labels_vector)
    features_vector = np.array(features_vector)
    
    reordered_preds, label_ordered = get_reverse_match(pred_vector, labels_vector)
    ece = calculate_ece(torch.tensor(con_vector).cpu(), torch.tensor(reordered_preds).cpu(), torch.tensor(labels_vector).cpu())*100
    #print("conf: ", np.mean(conf_hard))
    nmi, ari, acc = cluster_metric(labels_vector, pred_vector)
    
    if _indices is None:
        indices = get_top_indices_balanced(hard_vector, p=0.05)
        print(f"Indices of top 5% values: {len(indices), indices[:20]}")
    
    else:
        indices = _indices
        
    """ # 输出高置信样本的类别分布
    d_indices = get_top_5_percent_indices(hard_vector)
    top_targets = labels_vector[d_indices]
    class_counts = Counter(top_targets)

    print("Top 5% 样本的真实类别分布：")
    for cls, count in class_counts.items():
        print(f"类别 {cls}: {count} 个样本")
        
        
    #输出样本logits的类别分布
    logits = np.array(con_vector)  # [N, num_classes]
    targets = labels_vector    # [N]
    num_classes = logits.shape[1]

    avg_logits_per_class = []

    for c in range(num_classes):
        class_logits = logits[targets == c]
        if len(class_logits) > 0:
            avg_logit = class_logits.mean(axis=0)
        else:
            avg_logit = np.zeros(num_classes)
        avg_logits_per_class.append(avg_logit)

    avg_logits_per_class = np.stack(avg_logits_per_class)

    print("每个类别的平均logits：")
    for i, logit in enumerate(avg_logits_per_class):
        print(f"类别 {i}: {logit}") """
        
        
        
    #import pdb; pdb.set_trace()
    print("Evaluating hard samples……")
    labels_hard = labels_vector[indices]
    #pred_hard = pred_vector[indices]
    #nmi_h, ari_h, acc_h = cluster_metric(labels_hard, pred_hard)
    #print("Hard ACC, NMI, ARI:", acc_h, nmi_h, ari_h)

    #import pdb; pdb.set_trace()
    pred_adjusted = reorder_preds(pred_vector, labels_vector)
    pred_hard = pred_adjusted[indices]
    acc2 = metrics.accuracy_score(pred_hard, labels_hard)*100
    nmi2 = metrics.normalized_mutual_info_score(labels_hard, pred_hard)*100
    ari2 = metrics.adjusted_rand_score(labels_hard, pred_hard)*100
    
    reorder_preds_hard = reordered_preds[indices]
    conf_hard = np.array(con_vector)[indices]
    #print("hard conf: ", np.mean(conf_hard))
    ece2 = calculate_ece(torch.tensor(conf_hard).cpu(), torch.tensor(reorder_preds_hard).cpu(), torch.tensor(labels_hard).cpu())*100
    
    print("Hard ACC, NMI, ARI:", acc2, nmi2, ari2, ece2)
    
    if tsne:
        perform_tsne_analysis(features_vector, labels_vector, pred_adjusted, indices, path)
    
    if _indices is None:
        return {'ACC': round(acc,6),'NMI': round(nmi,6),
            'ARI': round(ari,6),'ECE': round(ece,6)}, {'HACC': round(acc2,6),'HNMI': round(nmi2,6),
            'HARI': round(ari2,6), 'HECE': round(ece2,6)}, indices
    else:
        return {'ACC': round(acc,6),'NMI': round(nmi,6),
            'ARI': round(ari,6),'ECE': round(ece,6)}, {'HACC': round(acc2,6),'HNMI': round(nmi2,6),
            'HARI': round(ari2,6), 'HECE': round(ece2,6)}
    
def save_top_images(dataloader, sorted_indices, top_k=20, save_path="top20_images.png"):
    """
    根据 sorted_indices 从 dataloader 中保存 top_k 张图像为一张图。

    参数:
        dataloader: PyTorch 的 DataLoader（ImageNet10）。
        sorted_indices: 排序后的全局索引列表。
        top_k: 要保存的图片数量（默认20）。
        save_path: 保存图像的文件路径。
    """
    # 汇总 dataloader 中的全部图像与标签
    all_images, all_labels = [], []
    for data in dataloader:
        images = data[0]
        labels = data[1]
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    #import pdb; pdb.set_trace()

    sorted_indices = sorted_indices[:top_k].copy()
    # 选取 top_k 张图像
    top_images = all_images[sorted_indices[:top_k]]
    top_labels = all_labels[sorted_indices[:top_k]]

    # 创建图像网格
    grid_img = torchvision.utils.make_grid(top_images, nrow=5, normalize=True, padding=2)
    npimg = grid_img.permute(1, 2, 0).cpu().numpy()

    # 保存图像
    plt.figure(figsize=(15, 8))
    plt.imshow(npimg)
    plt.title(f"Top {top_k} Images")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"图像已保存至: {os.path.abspath(save_path)}")
    
def save_bottom_images(dataloader, sorted_indices, top_k=20, save_path="top20_images.png"):
    """
    根据 sorted_indices 从 dataloader 中保存 top_k 张图像为一张图。

    参数:
        dataloader: PyTorch 的 DataLoader（ImageNet10）。
        sorted_indices: 排序后的全局索引列表。
        top_k: 要保存的图片数量（默认20）。
        save_path: 保存图像的文件路径。
    """
    # 汇总 dataloader 中的全部图像与标签
    all_images, all_labels = [], []
    for data in dataloader:
        images = data[0]
        labels = data[1]
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    #import pdb; pdb.set_trace()

    sorted_indices = sorted_indices[-top_k:].copy()
    # 选取 top_k 张图像
    top_images = all_images[sorted_indices]
    top_labels = all_labels[sorted_indices]

    # 创建图像网格
    grid_img = torchvision.utils.make_grid(top_images, nrow=5, normalize=True, padding=2)
    npimg = grid_img.permute(1, 2, 0).cpu().numpy()

    # 保存图像
    plt.figure(figsize=(15, 8))
    plt.imshow(npimg)
    plt.title(f"Top {top_k} Images")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"图像已保存至: {os.path.abspath(save_path)}")
    
def compute_confidence_distance_by_class(features, hard_vector, targets, interval_size=500):
    """
    计算每个类别每个置信度区间的：
    - 区间中心到类中心的距离
    - 区间平均置信度
    - 区间内样本与类中心距离的方差
    
    参数：
    - features: numpy array or torch tensor, shape = [N, D]
    - hard_vector: list or array of shape [N], 表示每个样本的置信度
    - targets: list or array of shape [N], 表示每个样本的类别
    - interval_size: 每个置信度区间的样本数
    
    返回：
    - stats_per_class: dict[class_id] -> list of tuples:
        (distance_to_center, mean_confidence, variance_to_center)
    """

    if not isinstance(features, np.ndarray):
        features = features.cpu().numpy()

    stats_per_class: Dict[int, List[tuple]] = {}

    class_ids = np.unique(targets)
    for cls in class_ids:
        cls_indices = np.where(targets == cls)[0]
        cls_confidences = hard_vector[cls_indices]
        cls_features = features[cls_indices]

        # 类中心
        class_center = np.mean(cls_features, axis=0)

        # 按置信度排序
        sorted_order = np.argsort(cls_confidences)
        sorted_cls_indices = cls_indices[sorted_order]
        sorted_cls_conf = cls_confidences[sorted_order]
        sorted_cls_features = features[sorted_cls_indices]

        stats = []
        num_samples = len(sorted_cls_indices)
        for start in range(0, num_samples, interval_size):
            end = min(start + interval_size, num_samples)
            if end - start < 2:
                continue  # 区间太小则跳过

            interval_feats = sorted_cls_features[start:end]
            interval_conf = sorted_cls_conf[start:end]

            interval_center = np.mean(interval_feats, axis=0)
            dist_to_center = np.linalg.norm(interval_center - class_center)
            mean_conf = np.mean(interval_conf)

            # 所有样本到类中心的距离，用于计算方差
            dists = np.linalg.norm(interval_feats - class_center, axis=1)
            var_dist = np.var(dists)

            stats.append((dist_to_center, mean_conf, var_dist))

            # 打印信息
            print(f"[Class {cls}] Interval {start}-{end} | "
                  f"Dist: {dist_to_center:.4f} | "
                  f"MeanConf: {mean_conf:.4f} | "
                  f"VarDist: {var_dist:.6f}")

        stats_per_class[int(cls)] = stats

    return stats_per_class
    
# get indices
def get_indices_in_range_balanced(hard_vector, targets=None, p_high=0.2, p_low=0.1, balance=False):
    """
    选择hard_vector在指定区间 [p_low, p_high] 的样本索引。
    如果 balance=True，则在每个类中分别进行选择。
    
    参数：
        hard_vector: List[float] or 1D Tensor，样本的难度值。
        targets: List[int]，每个样本的标签（当balance=True时必填）。
        p_high: float，选择的上界百分比（例如 0.2 表示前 20%）。
        p_low: float，选择的下界百分比（例如 0.1 表示前 10%）。
        balance: bool，是否在每个类别中分别选择。
    返回：
        selected_indices: List[int]，选中的样本索引。
    """
    n = len(hard_vector)

    if not balance:
        sorted_indices = sorted(range(n), key=lambda i: hard_vector[i])
        high_idx = int(n * (1 - p_high))
        low_idx = int(n * (1 - p_low))
        return sorted_indices[high_idx:low_idx]

    if targets is None:
        raise ValueError("targets cannot be None when balance=True")

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    selected_indices = []

    for cls, cls_indices in class_to_indices.items():
        cls_hardness = [hard_vector[i] for i in cls_indices]
        sorted_cls = sorted(zip(cls_indices, cls_hardness), key=lambda x: x[1])
        num_cls = len(cls_indices)
        high_idx = int(num_cls * (1 - p_high))
        low_idx = int(num_cls * (1 - p_low))
        range_cls = sorted_cls[high_idx:low_idx]
        top_cls = [idx for idx, _ in range_cls]
        selected_indices.extend(top_cls)

    return selected_indices


from collections import defaultdict

def get_indices_in_FlexRand_balanced(hard_vector, targets=None, p_high=0.9, p_low=0.1, balance=False, seed=42):
    """
    根据样本难度值划分为 Easy/Medium/Hard 三段：
    - [0, p_low): Easy -> 随机采样一半
    - [p_low, p_high): Medium -> 全部保留
    - [p_high, 1.0]: Hard -> 随机采样一半

    参数：
        hard_vector: List[float] or 1D Tensor，样本的难度值。
        targets: List[int]，样本标签（balance=True 时必填）。
        p_high: float，难度上界（如 0.9）
        p_low: float，难度下界（如 0.1）
        balance: bool，是否在每类内分别采样
        seed: int，随机种子
        
    返回：
        selected_indices: List[int]，选中的样本索引
    """
    random.seed(seed)
    n = len(hard_vector)

    if not balance:
        sorted_indices = sorted(range(n), key=lambda i: hard_vector[i])
        low_idx = int(n * p_low)
        high_idx = int(n * p_high)

        easy_indices = sorted_indices[:low_idx]
        medium_indices = sorted_indices[low_idx:high_idx]
        hard_indices = sorted_indices[high_idx:]

        num_easy = len(easy_indices) // 2
        num_hard = len(hard_indices) // 2

        sampled_easy = random.sample(easy_indices, num_easy)
        sampled_hard = random.sample(hard_indices, num_hard)

        selected_indices = sampled_easy + medium_indices + sampled_hard
        return selected_indices

    else:
        if targets is None:
            raise ValueError("targets cannot be None when balance=True")

        class_to_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            class_to_indices[label].append(idx)

        selected_indices = []

        for cls, cls_indices in class_to_indices.items():
            cls_hardness = [hard_vector[i] for i in cls_indices]
            sorted_cls = sorted(zip(cls_indices, cls_hardness), key=lambda x: x[1])

            num_cls = len(cls_indices)
            low_idx = int(num_cls * p_low)
            high_idx = int(num_cls * p_high)

            easy = [idx for idx, _ in sorted_cls[:low_idx]]
            medium = [idx for idx, _ in sorted_cls[low_idx:high_idx]]
            hard = [idx for idx, _ in sorted_cls[high_idx:]]

            num_easy = len(easy) // 2
            num_hard = len(hard) // 2

            sampled_easy = random.sample(easy, num_easy) if num_easy > 0 else []
            sampled_hard = random.sample(hard, num_hard) if num_hard > 0 else []

            selected_indices.extend(sampled_easy + medium + sampled_hard)

        return selected_indices


def get_top_indices_balanced(hard_vector, targets=None, p=0.1, balance=False):

    n = len(hard_vector)
    
    if not balance:
        num_samples = int(n * p)
        sorted_indices = sorted(range(n), key=lambda i: hard_vector[i])
        return sorted_indices[n - num_samples:]

    if targets is None:
        raise ValueError("targets cannot be None when balance=True")

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    selected_indices = []

    for cls, cls_indices in class_to_indices.items():
        cls_hardness = [hard_vector[i] for i in cls_indices]
        num_select = max(1, int(len(cls_indices) * p))
        sorted_cls = sorted(zip(cls_indices, cls_hardness), key=lambda x: x[1])
        top_cls = [idx for idx, _ in sorted_cls[-num_select:]]
        selected_indices.extend(top_cls)

    return selected_indices

def get_bottom_indices_balanced(hard_vector, targets=None, p=0.1, balance=False):
    """
    选择 bottom p% 的 easiest 样本索引。
    如果 balance=True，则每类分别选择 p% 的样本；否则全局选择。

    参数:
    - hard_vector: list or np.array，表示每个样本的难度值
    - targets: list or np.array，表示每个样本的类别标签（仅 balance=True 时需要）
    - p: float，表示选择比例（默认 5%）
    - balance: bool，是否对每类进行均衡采样

    返回:
    - selected_indices: list[int]，所选样本的索引
    """
    n = len(hard_vector)
    
    if not balance:
        num_samples = int(n * p)
        sorted_indices = sorted(range(n), key=lambda i: hard_vector[i])
        return sorted_indices[:num_samples]

    if targets is None:
        raise ValueError("targets cannot be None when balance=True")

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    selected_indices = []

    for cls, cls_indices in class_to_indices.items():
        cls_hardness = [hard_vector[i] for i in cls_indices]
        num_select = max(1, int(len(cls_indices) * p))
        sorted_cls = sorted(zip(cls_indices, cls_hardness), key=lambda x: x[1])
        bottom_cls = [idx for idx, _ in sorted_cls[:num_select]]
        selected_indices.extend(bottom_cls)

    return selected_indices

def get_indices_above_threshold(hard_vector, threshold=0.9):
    return [i for i, val in enumerate(hard_vector) if val > threshold]

def get_mid_5_percent_indices(hard_vector):
    n = len(hard_vector)
    num_samples = int(n * 0.05)  # 计算 5% 的样本数

    # 获取排序后的索引
    sorted_indices = sorted(range(n), key=lambda i: hard_vector[i])
    
    # 返回中间10%对应的索引
    return sorted_indices[n//2 - num_samples : n//2 + num_samples]

def plot_hard_clustering_metrics(path, acc_h_list, acc_list, 
                            nmi_h_list, nmi_list, 
                            ari_h_list, ari_list,
                            ece_h_list=None, ece_list=None):
    """
    绘制聚类指标（ACC, NMI, ARI, ECE）的对比曲线，分别对比 `_h` 版本和普通版本。

    参数：
    - acc_h_list: list，ACC_h（hard label 计算的 ACC）随 epoch 变化的列表
    - acc_list: list，ACC（普通 ACC）随 epoch 变化的列表
    - nmi_h_list: list，NMI_h（hard label 计算的 NMI）随 epoch 变化的列表
    - nmi_list: list，NMI（普通 NMI）随 epoch 变化的列表
    - ari_h_list: list，ARI_h（hard label 计算的 ARI）随 epoch 变化的列表
    - ari_list: list，ARI（普通 ARI）随 epoch 变化的列表
    - ece_h_list: list，ECE_h（hard label 计算的 ECE）随 epoch 变化的列表，默认为None
    - ece_list: list，ECE（普通 ECE）随 epoch 变化的列表，默认为None
    """
    import matplotlib.pyplot as plt
    import os

    epochs = list(range(1, len(acc_h_list) + 1))  # 生成 epoch 序列
    
    # 确定需要绘制的子图数量
    num_plots = 4 if (ece_h_list is not None and ece_list is not None) else 3
    
    # 创建对应行数的子图
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))

    # 绘制 ACC 对比曲线
    axes[0].plot(epochs, acc_h_list, marker='o', linestyle='-', label='ACC_h')
    axes[0].plot(epochs, acc_list, marker='s', linestyle='--', label='ACC')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('ACC Score')
    axes[0].set_title('Comparison of ACC_h and ACC')
    axes[0].legend()
    axes[0].grid(True)

    # 绘制 NMI 对比曲线
    axes[1].plot(epochs, nmi_h_list, marker='o', linestyle='-', label='NMI_h')
    axes[1].plot(epochs, nmi_list, marker='s', linestyle='--', label='NMI')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('NMI Score')
    axes[1].set_title('Comparison of NMI_h and NMI')
    axes[1].legend()
    axes[1].grid(True)

    # 绘制 ARI 对比曲线
    axes[2].plot(epochs, ari_h_list, marker='o', linestyle='-', label='ARI_h')
    axes[2].plot(epochs, ari_list, marker='s', linestyle='--', label='ARI')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('ARI Score')
    axes[2].set_title('Comparison of ARI_h and ARI')
    axes[2].legend()
    axes[2].grid(True)
    
    # 如果提供了ECE数据，绘制ECE对比曲线
    if ece_h_list is not None and ece_list is not None:
        axes[3].plot(epochs, ece_h_list, marker='o', linestyle='-', label='ECE_h')
        axes[3].plot(epochs, ece_list, marker='s', linestyle='--', label='ECE')
        axes[3].set_xlabel('Epochs')
        axes[3].set_ylabel('ECE Score')
        axes[3].set_title('Comparison of ECE_h and ECE')
        axes[3].legend()
        axes[3].grid(True)

    # 调整子图间距
    plt.tight_layout()
    save_path = os.path.join(path, "clustering_metrics.png")
    plt.savefig(save_path, dpi=300)
    print(f"图像已保存至 {save_path}")

def perform_tsne_analysis(features, labels, predictions, hard_indices, save_path):
    """
    对特征进行TSNE降维分析并可视化结果
    
    参数:
    - features: 高维特征向量 (n_samples, n_features)
    - labels: 真实标签 (n_samples,)
    - predictions: 预测标签 (n_samples,)
    - hard_indices: 困难样本索引
    - save_path: 图像保存路径
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import os
    
    print("正在执行TSNE降维分析...")
    
    # 应用TSNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(features)
    
    # 创建保存TSNE图像的目录
    tsne_dir = os.path.join(save_path, 'tsne_analysis')
    os.makedirs(tsne_dir, exist_ok=True)
    
    # 绘制按真实标签着色的TSNE图
    plt.figure(figsize=(12, 10))
    
    # 获取唯一的标签值
    unique_labels = np.unique(labels)
    
    # 用不同颜色绘制不同类别
    for label in unique_labels:
        plt.scatter(
            tsne_result[labels == label, 0],
            tsne_result[labels == label, 1],
            alpha=0.6,
            label=f'Class {label}'
        )
    
    # 用特殊标记绘制困难样本
    plt.scatter(
        tsne_result[hard_indices, 0],
        tsne_result[hard_indices, 1],
        c='red',
        marker='x',
        s=100,
        alpha=0.8,
        label='Hard Samples'
    )
    
    plt.title('TSNE Visualization of Features (True Labels)')
    plt.legend()
    plt.savefig(os.path.join(tsne_dir, 'tsne_true_labels.png'), dpi=300, bbox_inches='tight')
    
    # 绘制按预测标签着色的TSNE图
    plt.figure(figsize=(12, 10))
    
    # 获取唯一的预测值
    unique_predictions = np.unique(predictions)
    
    # 用不同颜色绘制不同预测类别
    for pred in unique_predictions:
        plt.scatter(
            tsne_result[predictions == pred, 0],
            tsne_result[predictions == pred, 1],
            alpha=0.6,
            label=f'Pred {pred}'
        )
    
    # 用特殊标记绘制困难样本
    plt.scatter(
        tsne_result[hard_indices, 0],
        tsne_result[hard_indices, 1],
        c='red',
        marker='x',
        s=100,
        alpha=0.8,
        label='Hard Samples'
    )
    
    plt.title('TSNE Visualization of Features (Predicted Labels)')
    plt.legend()
    plt.savefig(os.path.join(tsne_dir, 'tsne_predicted_labels.png'), dpi=300, bbox_inches='tight')
    
    # 绘制错误预测样本的TSNE图
    wrong_indices = np.where(predictions != labels)[0]
    
    plt.figure(figsize=(12, 10))
    # 绘制正确分类的样本
    plt.scatter(
        tsne_result[predictions == labels, 0],
        tsne_result[predictions == labels, 1],
        c='blue',
        alpha=0.5,
        label='Correct Predictions'
    )
    
    # 绘制错误分类的样本
    plt.scatter(
        tsne_result[wrong_indices, 0],
        tsne_result[wrong_indices, 1],
        c='orange',
        alpha=0.6,
        label='Wrong Predictions'
    )
    
    # 绘制困难样本
    plt.scatter(
        tsne_result[hard_indices, 0],
        tsne_result[hard_indices, 1],
        c='red',
        marker='x',
        s=100,
        alpha=0.8,
        label='Hard Samples'
    )
    
    plt.title('TSNE Visualization of Correct vs Wrong Predictions')
    plt.legend()
    plt.savefig(os.path.join(tsne_dir, 'tsne_errors.png'), dpi=300, bbox_inches='tight')
    
    print(f"TSNE分析完成。图像已保存到 {tsne_dir}")
    
    # 额外分析：计算困难样本与错误样本的重叠
    hard_set = set(hard_indices)
    wrong_set = set(wrong_indices)
    overlap = hard_set.intersection(wrong_set)
    
    print(f"困难样本总数: {len(hard_indices)}")
    print(f"错误样本总数: {len(wrong_indices)}")
    print(f"困难样本中的错误样本数: {len(overlap)}")
    print(f"困难样本中的错误率: {len(overlap)/len(hard_indices)*100:.2f}%")
    print(f"错误样本中是困难样本的比例: {len(overlap)/len(wrong_indices)*100:.2f}%")

def negative_outputs(all_logits_vector, index_vector, high_indices):
    all_logits = np.array(all_logits_vector)  # shape: [N, C]
    index_vector = np.array(index_vector)     # shape: [N]
    high_indices = set(high_indices)          # 转为 set 加速判断

    # 获取高置信样本的位置 mask
    high_mask = np.array([idx in high_indices for idx in index_vector])
    high_logits = all_logits[high_mask]        # [M, C]

    # 获取预测类别
    high_preds = np.argmax(high_logits, axis=1)  # [M]

    # 分类收集正输出和负输出（去除预测类）
    neg_logits_per_class = defaultdict(list)
    pos_logits_per_class = defaultdict(list)

    for logits, pred in zip(high_logits, high_preds):
        pred = int(pred)
        pos_logits_per_class[pred].append(logits[pred])                 # 正输出
        neg_logits = np.delete(logits, pred)                            # 负输出
        neg_logits_per_class[pred].append(neg_logits)

    print("\n========== 正输出高斯分布 ==========")
    for cls in sorted(pos_logits_per_class.keys()):
        pos_vals = np.array(pos_logits_per_class[cls])
        if len(pos_vals) > 0:
            mu_pos, std_pos = norm.fit(pos_vals)
            print(f"[Class {cls}] μ = {mu_pos:.4f}, σ = {std_pos:.4f}")
        else:
            print(f"[Class {cls}] 无正输出样本")

    print("\n========== 负输出高斯分布 ==========")
    for cls in sorted(neg_logits_per_class.keys()):
        neg_list = neg_logits_per_class[cls]
        if len(neg_list) > 0:
            all_neg = np.concatenate(neg_list)
            mu_neg, std_neg = norm.fit(all_neg)
            print(f"[Class {cls}] μ = {mu_neg:.4f}, σ = {std_neg:.4f}")
        else:
            print(f"[Class {cls}] 无负输出样本")





# divclust
def clustering_accuracy_metrics(cluster_labels, ground_truth):
    if isinstance(cluster_labels, torch.Tensor):
        cluster_labels = cluster_labels.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if len(cluster_labels.shape) == 1:
        cluster_labels = np.expand_dims(cluster_labels, 0)

    cluster_labels = cluster_labels.astype(np.int64)
    ground_truth = ground_truth.astype(np.int64)
    assert cluster_labels.shape[-1] == ground_truth.shape[-1]
    metrics = {}
    cluster_accuracies, cluster_nmis, cluster_aris = [], [], []
    interclustering_nmi = []
    clusterings = len(cluster_labels)
    
    # import pdb; pdb.set_trace()
    
    for k in range(clusterings):
        for j in range(clusterings):
            if j>k:
                interclustering_nmi.append(np.round(normalized_mutual_info_score(cluster_labels[k], cluster_labels[j]), 5))
        cluster_accuracies.append(clustering_acc(cluster_labels[k], ground_truth))
        cluster_nmis.append(np.round(normalized_mutual_info_score(cluster_labels[k], ground_truth), 5))
        cluster_aris.append(np.round(adjusted_rand_score(ground_truth, cluster_labels[k]), 5))
        metrics["cluster_acc_" + str(k)] = cluster_accuracies[-1]
        metrics["cluster_nmi_" + str(k)] = cluster_nmis[-1]
        metrics["cluster_ari_" + str(k)] = cluster_aris[-1]
    metrics["max_cluster_acc"], metrics["mean_cluster_acc"], metrics["min_cluster_acc"] = np.max(
        cluster_accuracies), np.mean(cluster_accuracies), np.min(cluster_accuracies)
    metrics["max_cluster_nmi"], metrics["mean_cluster_nmi"], metrics["min_cluster_nmi"] = np.max(
        cluster_nmis), np.mean(cluster_nmis), np.min(cluster_nmis)
    metrics["max_cluster_ari"], metrics["mean_cluster_ari"], metrics["min_cluster_ari"] = np.max(
        cluster_aris), np.mean(cluster_aris), np.min(cluster_aris)
    if clusterings>1:
        metrics["interclustering_nmi"] = sum(interclustering_nmi)/len(interclustering_nmi)
    return metrics

def clustering_acc(y_pred, y_true):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 100.0 / y_pred.size

# https://github.com/saandeepa93/FlowCon_OOD/blob/master/utils.py
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
def get_metrics_ood(label, score, invert_score=False):
    results_dict = {}
    if invert_score:
        score = score - score.max()
        score = np.abs(score)

    error = 1 - label
    rocauc = roc_auc_score(label, score)

    aupr_success = average_precision_score(label, score)
    aupr_errors = average_precision_score(error, (1 - score))

    precision, recall, thresholds = precision_recall_curve(label, score)

    # calculate fpr @ 95% tpr
    fpr = 0
    eval_range = np.arange(score.min(), score.max(), (score.max() - score.min()) / 10000)
    target_tpr = 0.95

    best_fpr = 100
    best_delta = score.min()
    for i, delta in enumerate(eval_range):
        tpr = len(score[(label == 1) & (score >= delta)]) / len(score[(label == 1)])
        fpr, tpr, thresholds = roc_curve(label, score)
        closest_tpr_index = np.argmin(np.abs(tpr - target_tpr))
        fpr = fpr[closest_tpr_index]
        if fpr < best_fpr:
            best_fpr = fpr
            best_delta = delta

        # if 0.9505 >= tpr >= 0.9495:
        #     fpr = len(score[(error == 1) & (score >= delta)]) / len(score[(error == 1)])
        #     print(delta)
        #     break

    print(best_fpr, best_delta)
    results_dict["rocauc"] = round(rocauc,4)*100
    results_dict["aupr_success"] = round(aupr_success,4)*100
    results_dict["aupr_error"] = round(aupr_errors,4)*100
    results_dict["fpr"] = round(best_fpr,4)*100
    return results_dict