import torch
import torch.nn.functional as F
import wandb
from cdc.utils.torch_clustering import PyTorchKMeans
from collections import Counter
from cdc.utils.evaluate_utils import get_predictions, hungarian_evaluate
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import random
import pdb
from typing import Tuple
import torch.nn.init as init
from sklearn.metrics import pairwise_distances

def orth_train(W, n_samples, scale = 5, epochs=2000, use_relu = False):
    Z = W.clone().cuda()
    #Z = Z.detach().clone()
    Z.requires_grad = True
    W_ = W.clone().cuda()
    #W_ = W_.detach().clone()
    W_.requires_grad = True
    labels = torch.arange(0, n_samples).cuda()
    optimizer = torch.optim.SGD([Z, W_], lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = torch.nn.CrossEntropyLoss()
    # with torch.enable_grad():
    for i in range(epochs):
        if use_relu:
            z = F.relu(Z)
        else:
            z = Z
        w = W_
        L2_z = F.normalize(z, dim=1)
        L2_w = F.normalize(w, dim=1)
        out = F.linear(L2_z, L2_w)
        loss = criterion(out * scale, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return W_.detach()

def initialize_weights(cfg, model, cali_mlp, features, val_dataloader):
    features_zscore = (features - features.mean(1).reshape(-1, 1)) / features.std(1).reshape(-1, 1)
    #features_zscore = features.detach()
    features_zscore = F.normalize(features_zscore, dim=1)

    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_zscore)
    W1 = KMeans_512.cluster_centers_

    #linear(512,512)
    H = torch.mm(features, W1.T)
    #BN
    # H = (H - H.mean(0)) / H.std(0)
    H = model.module.cluster_head[0][1](H).detach().clone()
    #relu
    # H = torch.nn.functional.relu(H)
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    #H_zscore = H.detach()
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_
    #print(W2[0])
    
    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, cfg['backbone']['nclusters'], use_relu=True)
    """ W1_modi = W1
    W2_modi = W2 """
    
    O = torch.mm(torch.mm(features, W1.T), W2.T)
    #print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    O = torch.mm(torch.mm(features, W1_modi.T) , W2_modi.T)
    #print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))

    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()

    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()


    # predictions = get_predictions(cfg, val_dataloader, model)
    # clustering_stats = hungarian_evaluate(cfg, cfg['cdc_checkpoint'], 0, 0,
    #                                     predictions, title=cfg['cluster_eval']['plot_title'],
    #                                     compute_confusion_matrix=False)
    # print(clustering_stats)

    

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_density_weights(features, labels, k=10, alpha=0.5, eps=1e-6):
    """
    计算每个样本的权重，密度高的区域权重低
    features: Tensor [N, D]  (可能在 GPU 上)
    labels: Tensor [N] (可能在 GPU 上)
    """
    device = features.device
    features_cpu = features.detach().cpu().numpy()
    labels_cpu = labels.detach().cpu().numpy()

    weights = np.zeros(len(features_cpu), dtype=np.float32)
    dis = np.zeros(len(features_cpu), dtype=np.float32)

    for c in np.unique(labels_cpu):
        idx = np.where(labels_cpu == c)[0]
        cluster_feats = features_cpu[idx]

        if len(idx) <= k:  # 簇太小，不做密度加权
            weights[idx] = 1.0
            dis[idx] = 1.0
            continue

        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(cluster_feats)
        distances, _ = nbrs.kneighbors(cluster_feats)
        # 去掉自己本身的距离（第一个是0）
        avg_dist = distances[:, 1:].mean(axis=1)

        #pdb.set_trace()

        cluster_weights = np.exp(alpha * avg_dist)
        cluster_weights = cluster_weights / (cluster_weights.sum() + eps)

        weights[idx] = cluster_weights
        dis[idx] = avg_dist

    return torch.tensor(weights, dtype=torch.float32, device=device)

def weighted_cluster_centers(features, labels, n_clusters, weights):
    """
    用权重重新计算簇中心
    """
    D = features.size(1)
    centers = torch.zeros((n_clusters, D), device=features.device)
    for c in range(n_clusters):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        #pdb.set_trace()
        if len(idx) > 0:
            w = weights[idx].unsqueeze(1)  # [Nc, 1]
            centers[c] = (features[idx] * w).sum(0) / (w.sum() + 1e-6)
    return centers

def train_cali(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, epoch, start_epoch):
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        optimizer_all.zero_grad()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        gt = batch['target'].cuda(non_blocking=True)

        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]

            feature_weak = model(images, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')

        feature_norm1 = F.normalize(feature_val, p=1, dim=1)
        clu_softmax = F.softmax(output_clu, dim=1)
        cali_softmax = F.softmax(output_cali, dim=1)
        clu_prob, clu_label = torch.max(clu_softmax, dim=1)
        cali_prob, cali_label = torch.max(cali_softmax, dim=1)

        proto_pseudo = cali_label
        selected_num = cfg['method_kwargs']['per_class_selected_num']
        # selected_num = int(output_cali.shape[0] / output_cali.shape[1])
        selected_idx = torch.zeros(len(cali_softmax)).cuda()
        for label_idx in range(output_clu.shape[1]):
            per_label_mask = cali_softmax[:, label_idx].sort(descending=True)[1][:selected_num]
            sel = int(cali_prob[per_label_mask].mean() * selected_num)
            selected_idx[per_label_mask[:sel]]=1
        selected_idx = selected_idx==1

        cluster_num = cfg['method_kwargs']['super_cluster_num']
        KMeans_all = PyTorchKMeans(init='k-means++', n_clusters=cluster_num, verbose=False)
        split_all = KMeans_all.fit_predict(feature_norm1)
        target_dict = torch.stack([F.softmax(output_clu_val, dim=1)[split_all == i].mean(0) for i in range(cluster_num)])
        super_target = target_dict[split_all]

        sub_steps = int(cfg['optimizer']['batch_size']/cfg['optimizer']['sub_batch_size'])
        sub_idxs = torch.range(0, sub_steps*cfg['optimizer']['sub_batch_size']-1).to(torch.int64).reshape(sub_steps,-1)
        for sub_step in range(sub_steps):
            sub_idx = sub_idxs[sub_step]
            output_aug = model(images_augmented[sub_idx])[0]
            sub_proto_pseudo, sub_selected_idx = proto_pseudo[sub_idx], selected_idx[sub_idx]
            loss_ce = F.cross_entropy(output_aug[sub_selected_idx], sub_proto_pseudo[sub_selected_idx])
            loss = loss_ce
            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss.detach())

            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()
            output_cali = cali_mlp(feature_val[sub_idx], forward_pass='calibration')
            cali_prob, _ = F.softmax(output_cali, dim=1).max(1)

            loss_cos = (-super_target[sub_idx]*F.log_softmax(output_cali)).sum(1).mean()
            x_ = torch.mean(F.softmax(output_cali, dim=1), 0)
            loss_entropy = torch.sum(x_ * torch.log(x_))

            loss = loss_cos+cfg['method_kwargs']['w_en']*loss_entropy

            loss_cali.append(loss.detach())
            loss_coss.append(loss_cos.detach())
            loss_ens.append(loss_entropy.detach())

            optimizer_cali.zero_grad()
            loss.backward()
            optimizer_cali.step()
            optimizer_cali.step()


    
    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })


def initialize_weights_bias(cfg, model, cali_mlp, features, val_dataloader, k=10, alpha=1.0):
    # 特征预处理
    features_zscore = (features - features.mean(1, keepdim=True)) / features.std(1, keepdim=True)
    features_zscore = F.normalize(features_zscore, dim=1)

    # Step1: 先KMeans 512
    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_zscore)
    proto_label = torch.tensor(proto_label, device=features.device)

    W1 = KMeans_512.cluster_centers_

    # Step2: 通过 cluster_head BN + ReLU
    H = torch.mm(features, W1.T)
    H = model.module.cluster_head[0][1](H).detach().clone()
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1, keepdim=True)) / H.std(1, keepdim=True)
    H_zscore = F.normalize(H_zscore, dim=1)

    # Step3: KMeans 最终类别数
    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    class_label = torch.tensor(class_label, device=features.device)

    # Step4: 用密度加权重新计算 W2
    density_weights2= compute_density_weights(H_zscore, class_label, k=k, alpha=alpha)
    density_weights2 = density_weights2 * (len(density_weights2) / (density_weights2.sum() + 1e-6))
    W2 = weighted_cluster_centers(H_zscore, class_label, cfg['backbone']['nclusters'], density_weights2)

    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, cfg['backbone']['nclusters'], use_relu=True)

    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()

    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()

    predictions = get_predictions(cfg, val_dataloader, model)
    clustering_stats = hungarian_evaluate(cfg, cfg['cdc_checkpoint'], 0, 0, predictions,
                                          title=cfg['cluster_eval']['plot_title'],
                                          compute_confusion_matrix=False)
    print(clustering_stats)


def initialize_weights_bias_v2(cfg, model, cali_mlp, features, val_dataloader, k=10, alpha=1.0, target_class= 3):
    # 特征预处理
    features_zscore = (features - features.mean(1, keepdim=True)) / features.std(1, keepdim=True)
    features_zscore = F.normalize(features_zscore, dim=1)

    # Step1: 先KMeans 512
    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_zscore)
    proto_label = torch.tensor(proto_label, device=features.device)

    # Step2: 用密度加权重新计算 W1
    density_weights = compute_density_weights(features_zscore, proto_label, k=int(k/2), alpha=alpha/2)
    density_weights = density_weights * (len(density_weights) / (density_weights.sum() + 1e-6))
    W1 = weighted_cluster_centers(features, proto_label, 512, density_weights)
    #W1 = KMeans_512.cluster_centers_

    # Step3: 通过 cluster_head BN + ReLU
    H = torch.mm(features, W1.T)
    H = model.module.cluster_head[0][1](H).detach().clone()
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1, keepdim=True)) / H.std(1, keepdim=True)
    H_zscore = F.normalize(H_zscore, dim=1)

    # Step4: KMeans 最终类别数
    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    class_label = torch.tensor(class_label, device=features.device)

    # Step5: 用密度加权重新计算 W2
    density_weights2 = compute_density_weights(H_zscore, class_label, k=k, alpha=alpha)
    density_weights2 = density_weights2 * (len(density_weights2) / (density_weights2.sum() + 1e-6))
    W2 = weighted_cluster_centers(H_zscore, class_label, cfg['backbone']['nclusters'], density_weights2)

    #W2 = KMeans_c.cluster_centers_

    density_weights2_np = density_weights2.detach().cpu().numpy()
    percentiles = np.percentile(density_weights2_np, [10, 40, 60, 90])
    indices_per_bin = []
    for i in range(5):
        if i == 0:
            mask = density_weights2_np <= percentiles[i]
        elif i == 4:
            mask = density_weights2_np > percentiles[i-1]
        else:
            mask = (density_weights2_np > percentiles[i-1]) & (density_weights2_np <= percentiles[i])

        # 进一步筛选出 target_class 的样本
        class_mask = (class_label == target_class).cpu().numpy().astype(bool)
        mask = mask & class_mask

        bin_indices = np.where(mask)[0]  # 该区间内目标类别的 index
        
        # 随机选取10个 index（如果不足10个就全取）
        chosen = np.random.choice(bin_indices, size=min(10, len(bin_indices)), replace=False)
        indices_per_bin.append(chosen)

    print(indices_per_bin)
    #pdb.set_trace()

    # Step6: 正交化（你原来的orth_train逻辑）
    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, cfg['backbone']['nclusters'], use_relu=True)

    # Step7: 写入 cluster_head 和 calibration_head
    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()

    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()

    # Step8: 做一次评估
    predictions = get_predictions(cfg, val_dataloader, model)
    clustering_stats = hungarian_evaluate(cfg, cfg['cdc_checkpoint'], 0, 0, predictions,
                                          title=cfg['cluster_eval']['plot_title'],
                                          compute_confusion_matrix=False)
    print(clustering_stats)

    return indices_per_bin

def visualize_bins(indices_per_bin, dataset, mean=(0.4914, 0.4823, 0.4466), std=(0.247, 0.243, 0.261), target_class=3):
    """
    indices_per_bin: list of numpy arrays, 每个 array 是一行
    dataset: 训练集 (支持 dataset[idx]['val'])
    mean, std: 用于还原的均值和标准差 (这里写的是 CIFAR-10 的)
    """
    nrows = len(indices_per_bin)
    ncols = max(len(arr) for arr in indices_per_bin)  # 每行列数 = 该行样本数量最大值

    fig, axes = plt.subplots(nrows, ncols, figsize=(2*ncols, 2*nrows))

    # 保证 axes 可迭代
    if nrows == 1:
        axes = [axes]

    for bin_idx, bin_indices in enumerate(indices_per_bin):
        for img_idx, idx in enumerate(bin_indices):
            img_tensor = dataset[idx]['val']  # 取图像
            img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,C]
            img = img * std + mean  # 反标准化
            img = img.clip(0, 1)

            plt.imshow(img)
            plt.title(f"Idx:{idx}", fontsize=8)
            plt.axis("off")

            # 每张图片单独保存
            plt.savefig(f"pic/stl10_class{target_class}_bin{bin_idx}_img{img_idx}_idx{idx}.png")
            plt.close()

    for row, bin_indices in enumerate(indices_per_bin):
        for col, idx in enumerate(bin_indices):
            img_tensor = dataset[idx]['val']  # 取图像
            img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,C]
            img = img * std + mean  # 反标准化
            img = img.clip(0, 1)

            axes[row][col].imshow(img)
            axes[row][col].set_title(f"Idx:{idx}", fontsize=8)
            axes[row][col].axis("off")

        # 如果该行比最大列数短，把空余子图关掉
        for col in range(len(bin_indices), ncols):
            axes[row][col].axis("off")

    plt.tight_layout()
    plt.savefig(f"pic/stl10_class{target_class}.png")

def show_image(img_tensor, title=None, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    img_tensor: torch.Tensor [C,H,W] (标准化后的图像)
    mean, std: 用于还原的均值和标准差 (这里写的是 CIFAR-10 的)
    """
    # 先转为 numpy
    img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)  # [H,W,C]

    # 反标准化 (还原到 0-1 区间)
    img = img * std + mean  
    img = img.clip(0, 1)

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.savefig("111.png")