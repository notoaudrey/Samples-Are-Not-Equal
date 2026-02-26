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

def initialize_weights_propos(cfg, model, cali_mlp, features, val_dataloader):
    
    features_zscore = (features - features.mean(1).reshape(-1, 1)) / features.std(1).reshape(-1, 1)
    features_zscore = F.normalize(features_zscore, dim=1)

    KMeans_256 = PyTorchKMeans(init='k-means++', n_clusters=256, verbose=False, random_state=0)
    proto_label = KMeans_256.fit_predict(features_zscore)
    W1 = KMeans_256.cluster_centers_

    #linear(512,512)
    H = torch.mm(features, W1.T)
    #BN
    # H = (H - H.mean(0)) / H.std(0)
    H = model.module.projector_classify[1](H).detach().clone()
    #relu
    # H = torch.nn.functional.relu(H)
    H = model.module.projector_classify[2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    #H_zscore = H.detach()
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_
    
    W1_modi = orth_train(W1, 256, use_relu=True)
    W2_modi = orth_train(W2, cfg['backbone']['nclusters'], use_relu=True)
    
    O = torch.mm(torch.mm(features, W1.T), W2.T)
    O = torch.mm(torch.mm(features, W1_modi.T) , W2_modi.T)

    torch.nn.init.zeros_(model.module.projector_classify[0].bias)
    torch.nn.init.zeros_(model.module.projector_classify[3].bias)
    model.module.projector_classify[0].weight.data = W1_modi.clone()
    model.module.projector_classify[3].weight.data = W2_modi.clone()

    torch.nn.init.zeros_(cali_mlp.module.calibration_propos[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_propos[3].bias)
    cali_mlp.module.calibration_propos[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_propos[3].weight.data = W2_modi.clone()


    predictions = get_predictions(cfg, val_dataloader, model, propos=True)
    clustering_stats = hungarian_evaluate(cfg, cfg['cdc_checkpoint'], 0, 0,
                                        predictions, title=cfg['cluster_eval']['plot_title'],
                                        compute_confusion_matrix=False)
    print(clustering_stats)

def init_head_with_confident_samples_propos(model, cali_mlp, features, predictions, n_clusters, top_ratio=0.5, confidence_offset=0.0, balanced_per_class=False):
    probs = torch.softmax(predictions, dim=1)
    confidences, pred_classes = torch.max(probs, dim=1)

    if balanced_per_class:
        selected_indices = []
        for c in range(n_clusters):
            cls_indices = (pred_classes == c).nonzero(as_tuple=False).squeeze()
            if cls_indices.numel() == 0:
                continue
            cls_confidences = confidences[cls_indices]
            sorted_indices = torch.argsort(cls_confidences, descending=True)
            offset = int(len(sorted_indices) * confidence_offset)
            top_k = max(1, int(len(sorted_indices) * top_ratio))
            end = min(offset + top_k, len(sorted_indices))
            selected_cls_idx = cls_indices[sorted_indices[offset:end]]
            selected_indices.append(selected_cls_idx)
        top_indices = torch.cat(selected_indices)
    else:
        sorted_indices = torch.argsort(confidences, descending=True)
        offset = int(len(sorted_indices) * confidence_offset)
        num_top = int(len(sorted_indices) * top_ratio)
        end = min(offset + num_top, len(sorted_indices))
        top_indices = sorted_indices[offset:end]

    features_sub = features[top_indices]
    print("ini sample: ", len(top_indices))
    
    features_zscore = (features_sub - features_sub.mean(1).reshape(-1, 1)) / features_sub.std(1).reshape(-1, 1)
    features_zscore = F.normalize(features_zscore, dim=1)

    KMeans_256 = PyTorchKMeans(init='k-means++', n_clusters=256, verbose=False, random_state=0)
    proto_label = KMeans_256.fit_predict(features_zscore)
    W1 = KMeans_256.cluster_centers_

    #linear(512,512)
    H = torch.mm(features, W1.T)
    #BN
    # H = (H - H.mean(0)) / H.std(0)
    H = model.module.projector_classify[1](H).detach().clone()
    #relu
    # H = torch.nn.functional.relu(H)
    H = model.module.projector_classify[2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    #H_zscore = H.detach()
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters= n_clusters, verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_
    
    W1_modi = orth_train(W1, 256, use_relu=True)
    W2_modi = orth_train(W2,  n_clusters, use_relu=True)
    
    O = torch.mm(torch.mm(features, W1.T), W2.T)
    O = torch.mm(torch.mm(features, W1_modi.T) , W2_modi.T)

    torch.nn.init.zeros_(model.module.projector_classify[0].bias)
    torch.nn.init.zeros_(model.module.projector_classify[3].bias)
    model.module.projector_classify[0].weight.data = W1_modi.clone()
    model.module.projector_classify[3].weight.data = W2_modi.clone()

    torch.nn.init.zeros_(cali_mlp.module.calibration_propos[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_propos[3].bias)
    cali_mlp.module.calibration_propos[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_propos[3].weight.data = W2_modi.clone()

    
    
def flexrand_select(features, predictions, top_ratio=0.4, gamma=0.3, balanced_per_class=True, seed=42):
    """
    FlexRand采样策略：在 Easy/Hard 区间中分别随机采样子集。
    
    参数:
        features: Tensor [N, D]，特征向量
        predictions: Tensor [N, C]，模型输出（logits）
        top_ratio: float，采样比例（最终总共选出 top_ratio * N 个样本）
        gamma: float，Easy 区间的比例（如 0.3）
        balanced_per_class: bool，是否在每类内使用 FlexRand
        seed: int，随机种子

    返回:
        features_sub: Tensor，采样后的特征子集
        top_indices: Tensor，对应的样本索引
    """
    random.seed(seed)
    probs = torch.softmax(predictions, dim=1)
    confidences, pred_classes = torch.max(probs, dim=1)
    n_clusters = predictions.shape[1]

    top_indices = []

    if balanced_per_class:
        for c in range(n_clusters):
            cls_indices = (pred_classes == c).nonzero(as_tuple=False).squeeze()
            if cls_indices.numel() == 0:
                continue

            cls_conf = confidences[cls_indices]
            sorted_idx = torch.argsort(cls_conf, descending=True)
            sorted_cls_indices = cls_indices[sorted_idx]

            num_total = len(sorted_cls_indices)
            num_select = int(num_total * top_ratio)
            if num_select < 2:
                continue  # 至少要能在两个区间中各取一个

            num_each = num_select // 2
            easy_end = int(num_total * gamma)
            hard_start = int(num_total * (1 - gamma))

            easy_pool = sorted_cls_indices[:easy_end]
            hard_pool = sorted_cls_indices[hard_start:]

            if len(easy_pool) >= num_each:
                sampled_easy = random.sample(easy_pool.tolist(), num_each)
            else:
                sampled_easy = easy_pool.tolist()

            if len(hard_pool) >= num_each:
                sampled_hard = random.sample(hard_pool.tolist(), num_each)
            else:
                sampled_hard = hard_pool.tolist()

            selected = sampled_easy + sampled_hard
            top_indices.extend(selected)
    else:
        sorted_idx = torch.argsort(confidences, descending=True)
        num_total = len(sorted_idx)
        num_select = int(num_total * top_ratio)
        num_each = num_select // 2
        easy_end = int(num_total * gamma)
        hard_start = int(num_total * (1 - gamma))

        easy_pool = sorted_idx[:easy_end]
        hard_pool = sorted_idx[hard_start:]

        sampled_easy = random.sample(easy_pool.tolist(), num_each)
        sampled_hard = random.sample(hard_pool.tolist(), num_each)

        top_indices = sampled_easy + sampled_hard

    top_indices = torch.tensor(top_indices, dtype=torch.long).sort().values
    features_sub = features[top_indices]
    print("FlexRand sample:", len(top_indices))

    return features_sub, top_indices    
    
def select_flexrand_middle_all(
    features: torch.Tensor,
    predictions: torch.Tensor,
    p_low: float = 0.1,
    p_high: float = 0.9,
    easy_ratio: float = 0.3,
    hard_ratio: float = 0.3,
    balanced_per_class: bool = True,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FlexRand with full middle + partial easy/hard samples

    参数:
        features: Tensor [N, D]
        predictions: Tensor [N, C]
        p_low: float, e.g., 0.1 (bottom 10% is easy)
        p_high: float, e.g., 0.9 (top 10% is hard)
        easy_ratio: float, ratio to sample from easy region
        hard_ratio: float, ratio to sample from hard region
        balanced_per_class: bool, whether to sample per class
        seed: int, random seed

    返回:
        features_sub: 被选中特征
        selected_indices: 样本索引
    """
    random.seed(seed)
    torch.manual_seed(seed)

    probs = torch.softmax(predictions, dim=1)
    confidences, pred_classes = torch.max(probs, dim=1)
    n_clusters = predictions.shape[1]

    selected_indices = []

    if balanced_per_class:
        for c in range(n_clusters):
            cls_indices = (pred_classes == c).nonzero(as_tuple=False).squeeze()
            if cls_indices.numel() == 0:
                continue

            cls_conf = confidences[cls_indices]
            sorted_idx = torch.argsort(cls_conf, descending=True)
            sorted_cls_indices = cls_indices[sorted_idx]

            n_cls = len(sorted_cls_indices)
            low_idx = int(n_cls * p_low)
            high_idx = int(n_cls * p_high)

            easy_pool = sorted_cls_indices[:low_idx]
            middle_pool = sorted_cls_indices[low_idx:high_idx]
            hard_pool = sorted_cls_indices[high_idx:]

            num_easy = int(len(easy_pool) * easy_ratio)
            num_hard = int(len(hard_pool) * hard_ratio)

            sampled_easy = random.sample(easy_pool.tolist(), min(num_easy, len(easy_pool))) if len(easy_pool) > 0 else []
            sampled_hard = random.sample(hard_pool.tolist(), min(num_hard, len(hard_pool))) if len(hard_pool) > 0 else []

            selected = sampled_easy + middle_pool.tolist() + sampled_hard
            selected_indices.extend(selected)
    else:
        sorted_idx = torch.argsort(confidences, descending=True)
        n_total = len(sorted_idx)
        low_idx = int(n_total * p_low)
        high_idx = int(n_total * p_high)

        easy_pool = sorted_idx[:low_idx]
        middle_pool = sorted_idx[low_idx:high_idx]
        hard_pool = sorted_idx[high_idx:]

        num_easy = int(len(easy_pool) * easy_ratio)
        num_hard = int(len(hard_pool) * hard_ratio)

        sampled_easy = random.sample(easy_pool.tolist(), min(num_easy, len(easy_pool))) if len(easy_pool) > 0 else []
        sampled_hard = random.sample(hard_pool.tolist(), min(num_hard, len(hard_pool))) if len(hard_pool) > 0 else []

        selected_indices = sampled_easy + middle_pool.tolist() + sampled_hard

    selected_indices = torch.tensor(selected_indices, dtype=torch.long).sort().values
    features_sub = features[selected_indices]
    print(f"Selected samples: Easy~{easy_ratio}, Middle~ALL, Hard~{hard_ratio} -> Total: {len(selected_indices)}")

    return features_sub, selected_indices
    
def visualize_tsne_kmeans(features, cluster_labels, selected_indices, title="t-SNE of KMeans results"):
    """
    t-SNE 可视化 KMeans 聚类结果，并标出被去掉的样本
    """
    # 转 numpy
    features_np = features.detach().cpu().numpy()
    if isinstance(cluster_labels, torch.Tensor):
        cluster_labels = cluster_labels.detach().cpu().numpy()
    else:
        cluster_labels = np.array(cluster_labels)

    tsne = TSNE(n_components=2, random_state=0, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(features_np)

    selected_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
    selected_mask[selected_indices] = True
    selected_mask_np = selected_mask.cpu().numpy()
    dropped_mask_np = ~selected_mask_np

    plt.figure(figsize=(8, 8))

    # 已选样本：按聚类标签上色
    plt.scatter(features_2d[selected_mask_np, 0], features_2d[selected_mask_np, 1],
                c=cluster_labels[selected_mask_np], cmap="tab10", s=10, alpha=0.7, label="Selected")

    # 被去掉样本：灰色叉号
    plt.scatter(features_2d[dropped_mask_np, 0], features_2d[dropped_mask_np, 1],
                c="lightgray", marker="x", s=20, alpha=0.6, label="Dropped")

    plt.title(title)
    plt.legend()
    plt.savefig("tsne_kmeans_visualization.png")  

def init_head_with_confident_samples(model, cali_mlp, features, predictions, n_clusters, top_ratio=0.5, confidence_offset=0.0, balanced_per_class=False):
    """
    用高置信度样本做 KMeans 初始化聚类头
    参数:
        model: 模型对象，包含 cluster_head 和 classify_tail
        features: Tensor[N, D]，所有样本特征
        predictions: Tensor[N, C]，每个样本的 softmax logits
        n_clusters: 类别数
        top_ratio: 每类或全局选择的比例
        balanced_per_class: 是否每个类单独选 top 样本（推荐）

    返回:
        None（直接修改 model.cluster_head 和 classify_tail）
    """
    probs = torch.softmax(predictions, dim=1)
    confidences, pred_classes = torch.max(probs, dim=1)

    if balanced_per_class:
        selected_indices = []
        for c in range(n_clusters):
            cls_indices = (pred_classes == c).nonzero(as_tuple=False).squeeze()
            if cls_indices.numel() == 0:
                continue
            cls_confidences = confidences[cls_indices]
            # 排序后选取 offset -> offset + top_k
            sorted_indices = torch.argsort(cls_confidences, descending=True)
            offset = int(len(sorted_indices) * confidence_offset)
            top_k = max(1, int(len(sorted_indices) * top_ratio))
            end = min(offset + top_k, len(sorted_indices))
            selected_cls_idx = cls_indices[sorted_indices[offset:end]]

            selected_indices.append(selected_cls_idx)
        top_indices = torch.cat(selected_indices)
    else:
        sorted_indices = torch.argsort(confidences, descending=True)
        offset = int(len(sorted_indices) * confidence_offset)
        num_top = int(len(sorted_indices) * top_ratio)
        end = min(offset + num_top, len(sorted_indices))
        top_indices = sorted_indices[offset:end]

    # 特征处理
    #features_sub = features[top_indices]
    #features_sub = features
    #top_indices = top_indices.sort().values
    
    """ features_sub, top_indices = select_flexrand_middle_all(
        features=features,
        predictions=predictions,
        p_low=0.1,
        p_high=0.9,
        easy_ratio=0.5,
        hard_ratio=0.5,
        balanced_per_class=True
    ) """


    features_sub = features[top_indices]
    print("ini sample: ", len(top_indices))
    
    features_sub = (features_sub - features_sub.mean(1, keepdim=True)) / (features_sub.std(1, keepdim=True) + 1e-6)
    #features_sub = (features_sub - features_sub.mean(1).reshape(-1, 1)) / (features_sub.std(1).reshape(-1, 1))
    features_sub = F.normalize(features_sub, dim=1)
    
    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_sub)
    W1 = KMeans_512.cluster_centers_

    #linear(512,512)
    H = torch.mm(features, W1.T)
    #H = torch.mm(features_sub, W1.T)
    #BN
    # H = (H - H.mean(0)) / H.std(0)
    H = model.module.cluster_head[0][1](H).detach().clone()
    #relu
    # H = torch.nn.functional.relu(H)
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    #H_zscore = H.detach()
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=n_clusters, verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_

    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, n_clusters, use_relu=True)
    
    O = torch.mm(torch.mm(features, W1.T), W2.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    O = torch.mm(torch.mm(features, W1_modi.T) , W2_modi.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))

    with torch.no_grad():
        torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
        torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
        
        model.module.cluster_head[0][0].weight.data = W1_modi.clone()
        model.module.cluster_head[0][3].weight.data = W2_modi.clone()

        torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
        torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
        
        cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
        cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()
        
    #visualize_tsne_kmeans(features, class_label, top_indices)
    
def init_head_with_logits_samples(model, cali_mlp, features, predictions, n_clusters, top_ratio=0.5, confidence_offset=0.0, balanced_per_class=False):
    """
    用高置信度样本做 KMeans 初始化聚类头
    参数:
        model: 模型对象，包含 cluster_head 和 classify_tail
        features: Tensor[N, D]，所有样本特征
        predictions: Tensor[N, C]，每个样本的 softmax logits
        n_clusters: 类别数
        top_ratio: 每类或全局选择的比例
        balanced_per_class: 是否每个类单独选 top 样本（推荐）

    返回:
        None(直接修改 model.cluster_head 和 classify_tail)
    """
    probs = predictions
    confidences, pred_classes = torch.max(probs, dim=1)


    if balanced_per_class:
        selected_indices = []
        for c in range(n_clusters):
            cls_indices = (pred_classes == c).nonzero(as_tuple=False).squeeze()
            if cls_indices.numel() == 0:
                continue
            cls_confidences = confidences[cls_indices]
            # 排序后选取 offset -> offset + top_k
            sorted_indices = torch.argsort(cls_confidences, descending=True)
            offset = int(len(sorted_indices) * confidence_offset)
            top_k = max(1, int(len(sorted_indices) * top_ratio))
            end = min(offset + top_k, len(sorted_indices))
            selected_cls_idx = cls_indices[sorted_indices[offset:end]]

            selected_indices.append(selected_cls_idx)
        top_indices = torch.cat(selected_indices)
    else:
        sorted_indices = torch.argsort(confidences, descending=True)
        offset = int(len(sorted_indices) * confidence_offset)
        num_top = int(len(sorted_indices) * top_ratio)
        end = min(offset + num_top, len(sorted_indices))
        top_indices = sorted_indices[offset:end]

    # 特征处理
    #features_sub = features[top_indices]
    #features_sub = features
    #top_indices = top_indices.sort().values
    
    """ features_sub, top_indices = select_flexrand_middle_all(
        features=features,
        predictions=predictions,
        p_low=0.1,
        p_high=0.9,
        easy_ratio=0.5,
        hard_ratio=0.5,
        balanced_per_class=True
    ) """


    features_sub = features[top_indices]
    print("ini sample: ", len(top_indices))
    
    features_sub = (features_sub - features_sub.mean(1, keepdim=True)) / (features_sub.std(1, keepdim=True) + 1e-6)
    #features_sub = (features_sub - features_sub.mean(1).reshape(-1, 1)) / (features_sub.std(1).reshape(-1, 1))
    features_sub = F.normalize(features_sub, dim=1)
    
    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_sub)
    W1 = KMeans_512.cluster_centers_

    #linear(512,512)
    H = torch.mm(features, W1.T)
    #H = torch.mm(features_sub, W1.T)
    #BN
    # H = (H - H.mean(0)) / H.std(0)
    H = model.module.cluster_head[0][1](H).detach().clone()
    #relu
    # H = torch.nn.functional.relu(H)
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    #H_zscore = H.detach()
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=n_clusters, verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_

    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, n_clusters, use_relu=True)
    
    O = torch.mm(torch.mm(features, W1.T), W2.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    O = torch.mm(torch.mm(features, W1_modi.T) , W2_modi.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))

    with torch.no_grad():
        torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
        torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
        
        model.module.cluster_head[0][0].weight.data = W1_modi.clone()
        model.module.cluster_head[0][3].weight.data = W2_modi.clone()

        torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
        torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
        
        cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
        cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()
     
    #visualize_tsne_kmeans(features, class_label, top_indices)
    
def init_head_with_real_logits_samples(model, cali_mlp, features, predictions, n_clusters, top_ratio=0.5, confidence_offset=0.0, balanced_per_class=False):
    """
    用高置信度样本做 KMeans 初始化聚类头
    参数:
        model: 模型对象，包含 cluster_head 和 classify_tail
        features: Tensor[N, D]，所有样本特征
        predictions: Tensor[N, C]，每个样本的 softmax logits
        n_clusters: 类别数
        top_ratio: 每类或全局选择的比例
        balanced_per_class: 是否每个类单独选 top 样本（推荐）

    返回:
        None(直接修改 model.cluster_head 和 classify_tail)
    """
    
    probs = predictions
    confidences, pred_classes = torch.max(probs, dim=1)

    #import pdb; pdb.set_trace()

    if balanced_per_class:
        selected_indices = []
        for c in range(n_clusters):
            cls_indices = (pred_classes == c).nonzero(as_tuple=False).squeeze()
            if cls_indices.numel() == 0:
                continue
            cls_confidences = confidences[cls_indices]
            # 排序后选取 offset -> offset + top_k
            sorted_indices = torch.argsort(cls_confidences, descending=True)
            offset = int(len(sorted_indices) * confidence_offset)
            top_k = max(1, int(len(sorted_indices) * top_ratio))
            end = min(offset + top_k, len(sorted_indices))
            selected_cls_idx = cls_indices[sorted_indices[offset:end]]

            selected_indices.append(selected_cls_idx)
        top_indices = torch.cat(selected_indices)
    else:
        sorted_indices = torch.argsort(confidences, descending=True)
        offset = int(len(sorted_indices) * confidence_offset)
        num_top = int(len(sorted_indices) * top_ratio)
        end = min(offset + num_top, len(sorted_indices))
        top_indices = sorted_indices[offset:end]

    # 特征处理
    #features_sub = features[top_indices]
    #features_sub = features
    #top_indices = top_indices.sort().values
    
    """ features_sub, top_indices = select_flexrand_middle_all(
        features=features,
        predictions=predictions,
        p_low=0.1,
        p_high=0.9,
        easy_ratio=0.5,
        hard_ratio=0.5,
        balanced_per_class=True
    ) """


    features_sub = features[top_indices]
    print("ini sample: ", len(top_indices))
    
    features_sub = (features_sub - features_sub.mean(1, keepdim=True)) / (features_sub.std(1, keepdim=True) + 1e-6)
    #features_sub = (features_sub - features_sub.mean(1).reshape(-1, 1)) / (features_sub.std(1).reshape(-1, 1))
    features_sub = F.normalize(features_sub, dim=1)
    
    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_sub)
    W1 = KMeans_512.cluster_centers_

    #linear(512,512)
    H = torch.mm(features, W1.T)
    #H = torch.mm(features_sub, W1.T)
    #BN
    # H = (H - H.mean(0)) / H.std(0)
    H = model.module.cluster_head[0][1](H).detach().clone()
    #relu
    # H = torch.nn.functional.relu(H)
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    #H_zscore = H.detach()
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=n_clusters, verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_

    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, n_clusters, use_relu=True)
    
    O = torch.mm(torch.mm(features, W1.T), W2.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    O = torch.mm(torch.mm(features, W1_modi.T) , W2_modi.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))

    with torch.no_grad():
        torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
        torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
        
        model.module.cluster_head[0][0].weight.data = W1_modi.clone()
        model.module.cluster_head[0][3].weight.data = W2_modi.clone()

        torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
        torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
        
        cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
        cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()
     
    visualize_tsne_kmeans(features, class_label, top_indices)

import time  
def train_cali(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, epoch, start_epoch):
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    epoch_start = time.time()   # 开始计时

    time_dataloader, time_forward, time_loss, time_backward, time_step = 0, 0, 0, 0, 0

    add_time = time.time()
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        optimizer_all.zero_grad()
        st = time.time()

        batch_start = time.perf_counter()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        gt = batch['target'].cuda(non_blocking=True)

        torch.cuda.synchronize()
        start_forward = time.perf_counter()

        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]

            feature_weak = model(images, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')
        

        torch.cuda.synchronize()
        time_forward += time.perf_counter() - start_forward

        # ------------------- Loss 准备 -------------------
        torch.cuda.synchronize()
        start_loss = time.perf_counter()

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

        torch.cuda.synchronize()
        time_loss += time.perf_counter() - start_loss

        sub_steps = int(cfg['optimizer']['batch_size']/cfg['optimizer']['sub_batch_size'])
        sub_idxs = torch.range(0, sub_steps*cfg['optimizer']['sub_batch_size']-1).to(torch.int64).reshape(sub_steps,-1)
        for sub_step in range(sub_steps):
            sub_idx = sub_idxs[sub_step]
            


            # ------------------- Forward -------------------
            torch.cuda.synchronize()
            start_forward2 = time.perf_counter()
            output_aug = model(images_augmented[sub_idx])[0]
            torch.cuda.synchronize()
            time_forward += time.perf_counter() - start_forward2

            torch.cuda.synchronize()
            start_loss2 = time.perf_counter()
            sub_proto_pseudo, sub_selected_idx = proto_pseudo[sub_idx], selected_idx[sub_idx]
            loss_ce = F.cross_entropy(output_aug[sub_selected_idx], sub_proto_pseudo[sub_selected_idx])
            loss = loss_ce
            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss.detach())

            torch.cuda.synchronize()
            time_loss += time.perf_counter() - start_loss2

            optimizer_all.zero_grad()

            torch.cuda.synchronize()
            start_backward = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            time_backward += time.perf_counter() - start_backward

            torch.cuda.synchronize()
            start_step = time.perf_counter()
            optimizer_all.step()
            torch.cuda.synchronize()
            time_step += time.perf_counter() - start_step


            torch.cuda.synchronize()
            start_forward3 = time.perf_counter()
            output_cali = cali_mlp(feature_val[sub_idx], forward_pass='calibration')
            cali_prob, _ = F.softmax(output_cali, dim=1).max(1)
            torch.cuda.synchronize()
            time_forward += time.perf_counter() - start_forward3

            torch.cuda.synchronize()
            start_loss3 = time.perf_counter()
            loss_cos = (-super_target[sub_idx]*F.log_softmax(output_cali)).sum(1).mean()
            x_ = torch.mean(F.softmax(output_cali, dim=1), 0)
            loss_entropy = torch.sum(x_ * torch.log(x_))

            loss = loss_cos+cfg['method_kwargs']['w_en']*loss_entropy

            loss_cali.append(loss.detach())
            loss_coss.append(loss_cos.detach())
            loss_ens.append(loss_entropy.detach())

            torch.cuda.synchronize()
            time_loss += time.perf_counter() - start_loss3

            optimizer_cali.zero_grad()
            torch.cuda.synchronize()
            start_backward2 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            time_backward += time.perf_counter() - start_backward2

            torch.cuda.synchronize()
            start_step2 = time.perf_counter()
            optimizer_cali.step()
            torch.cuda.synchronize()
            time_step += time.perf_counter() - start_step2

            optimizer_cali.step()

    epoch_time = time.time() - epoch_start

    print(f"[Time] DataLoader: {time_dataloader:.3f}s, "
          f"Forward: {time_forward:.3f}s, "
          f"Loss: {time_loss:.3f}s, "
          f"Backward: {time_backward:.3f}s, "
          f"Step: {time_step:.3f}s, "
          f"Total epoch: {epoch_time:.3f}s")
    
    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })
    print("epoch_time: ", epoch_time)
    return epoch_time

def train_cali_propos(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, epoch, start_epoch):
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        optimizer_all.zero_grad()
        import time
        st = time.time()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        gt = batch['target'].cuda(non_blocking=True)

        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone_propos')
            output_clu_val = model(feature_val, forward_pass='head_propos')[0]

            feature_weak = model(images, forward_pass='backbone_propos')
            output_clu = model(feature_weak, forward_pass='head_propos')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration_propos')

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

            output_cali = cali_mlp(feature_val[sub_idx], forward_pass='calibration_propos')
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
    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })

def train_cali_longtail(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, epoch, start_epoch, pseudo_labels, medium_neighbors_idx, tail_neighbors_idx):
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    epsilon = cfg['epsilon']
    num_classes = cfg['backbone']['nclusters']
    
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        optimizer_all.zero_grad()
        import time
        st = time.time()
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
            #output_tail = model.module.classify_tail(feature_weak)
            #output_medium = model.module.classify_medium(feature_weak)
            output_tail = output_clu
            output_medium = output_clu
            
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
            sub_batch_size = sub_idx.shape[0]
            output_aug = model(images_augmented[sub_idx])[0]
            sub_proto_pseudo, sub_selected_idx = proto_pseudo[sub_idx], selected_idx[sub_idx]
            
            # soft targets for tail / medium expert
            sub_indices = images_index[sub_idx]  # 当前子batch的全局样本 index
            sub_pseudo = sub_proto_pseudo  # 预测伪标签
                # 初始化 one-hot 平滑标签
            target_tail = torch.zeros(len(sub_pseudo), num_classes).cuda()
            target_medium = torch.zeros(len(sub_pseudo), num_classes).cuda()
            
            for j in range(len(sub_pseudo)):
                label = sub_pseudo[j].item()
                target_tail[j, label] = 1 - epsilon
                target_medium[j, label] = 1 - epsilon
                    # 获取该样本的邻居标签
                idx_j = sub_indices[j].item()
                tail_neigh = tail_neighbors_idx[idx_j]
                medium_neigh = medium_neighbors_idx[idx_j]

                if len(tail_neigh) > 0:
                    target_tail[j, tail_neigh] += epsilon / len(tail_neigh)
                if len(medium_neigh) > 0:
                    target_medium[j, medium_neigh] += epsilon / len(medium_neigh)
                
            target_tail = target_tail / target_tail.sum(dim=1, keepdim=True)
            target_medium = target_medium / target_medium.sum(dim=1, keepdim=True)

            spc_dict = Counter(pseudo_labels.cpu().numpy())  # {cluster_id: count}
            n_clusters = cfg['backbone']['nclusters']

            # 构建 spc 列表，并确保最小值为1（避免 log(0)）
            spc = [spc_dict.get(i, 1) for i in range(n_clusters)]
            # 转换为 Tensor 并放到 CUDA
            spc = torch.tensor(spc, dtype=torch.float32).cuda() 
            
            output_tail_sub = output_tail[sub_idx]
            output_medium_sub = output_medium[sub_idx]
            output_tail_sel = output_tail_sub[sub_selected_idx]
            output_medium_sel = output_medium_sub[sub_selected_idx]
            target_tail_sel = target_tail[sub_selected_idx]
            target_medium_sel = target_medium[sub_selected_idx]
            adj_tail = output_tail_sel + 1.0 * spc.log()
            adj_medium = output_medium_sel + 0.5 * spc.log()


            loss_tail = -torch.sum(F.log_softmax(adj_tail, dim=1) * target_tail_sel) /  target_tail_sel.shape[0]
            loss_medium = -torch.sum(F.log_softmax(adj_medium, dim=1) * target_medium_sel) /  target_medium_sel.shape[0]
            
            loss_ce = F.cross_entropy(output_aug[sub_selected_idx], sub_proto_pseudo[sub_selected_idx])
            loss = loss_ce +  loss_tail +  loss_medium
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
    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })
      
def initialize_weights_v4(cfg, model, cali_mlp, features, top_k_percent=0.5):
    print('Initializing weights V4 with high confidence samples...')
    features_zscore = (features - features.mean(1).reshape(-1, 1)) / features.std(1).reshape(-1, 1)
    features_zscore = F.normalize(features_zscore, dim=1)

    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_zscore)
    W1_initial = KMeans_512.cluster_centers_

    H = torch.mm(features, W1_initial.T)
    H = model.module.cluster_head[0][1](H).detach().clone() # BN
    H = model.module.cluster_head[0][2](H).detach().clone() # ReLU
    
    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    H_zscore = F.normalize(H_zscore, dim=1)
    
    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    W2_initial = KMeans_c.cluster_centers_
    
    # 计算第一阶段的样本到其所属原型簇中心的距离平方
    distances_to_proto_centers = torch.sum((features_zscore - W1_initial[proto_label])**2, dim=1)
    
    # 计算第二阶段的样本到其所属最终簇中心的距离平方
    distances_to_class_centers = torch.sum((H_zscore - W2_initial[class_label])**2, dim=1)
    
    high_conf_indices_1 = []
    for i in range(512):
        # 获取当前原型簇的所有样本索引
        cluster_samples_indices_tuple = (proto_label == i).nonzero(as_tuple=True)[0]
        # 修正：检查元组中的张量是否为空
        if len(cluster_samples_indices_tuple) == 0:
            continue

        # 修正：从元组中取出实际的索引张量
        current_cluster_indices = cluster_samples_indices_tuple 

        # 获取这些样本的距离
        cluster_distances = distances_to_proto_centers[current_cluster_indices]

        # 找到距离最小的 top_k_percent 样本的索引
        num_to_select = max(1, int(len(current_cluster_indices) * top_k_percent)) # 使用张量的长度
        sorted_indices = torch.argsort(cluster_distances)[:num_to_select]

        # 修正：正确索引张量
        high_conf_indices_1.extend(current_cluster_indices[sorted_indices].tolist())

    high_conf_indices_1 = torch.tensor(high_conf_indices_1).cuda()
    
    high_conf_indices_2 =[]
    for i in range(cfg['backbone']['nclusters']):
        # 获取当前最终簇的所有样本索引
        cluster_samples_indices_tuple = (class_label == i).nonzero(as_tuple=True)[0]
        # 修正：检查元组中的张量是否为空
        if len(cluster_samples_indices_tuple) == 0:
            continue

        # 修正：从元组中取出实际的索引张量
        current_cluster_indices = cluster_samples_indices_tuple

        # 获取这些样本的距离
        cluster_distances = distances_to_class_centers[current_cluster_indices]

        # 找到距离最小的 top_k_percent 样本的索引
        num_to_select = max(1, int(len(current_cluster_indices) * top_k_percent)) # 使用张量的长度
        sorted_indices = torch.argsort(cluster_distances)[:num_to_select]

        # 修正：正确索引张量
        high_conf_indices_2.extend(current_cluster_indices[sorted_indices].tolist())

    high_conf_indices_2 = torch.tensor(high_conf_indices_2).cuda()
    
    W1_modi = torch.zeros_like(W1_initial)
    for i in range(512):
        # 筛选出属于当前簇且在高置信度列表中的样本
        current_cluster_high_conf_samples = features_zscore[
            (proto_label == i) & (torch.isin(torch.arange(len(features_zscore), device=features.device), high_conf_indices_1))
        ]
        if len(current_cluster_high_conf_samples) > 0:
            W1_modi[i] = current_cluster_high_conf_samples.mean(dim=0)
        else:
            W1_modi[i] = W1_initial[i] # 兜底：如果筛选后无样本，则使用原始中心
            
    W2_modi = torch.zeros_like(W2_initial)
    for i in range(cfg['backbone']['nclusters']):
        # 筛选出属于当前簇且在高置信度列表中的样本
        current_cluster_high_conf_samples = H_zscore[
            (class_label == i) & (torch.isin(torch.arange(len(H_zscore), device=H_zscore.device), high_conf_indices_2))
        ]
        if len(current_cluster_high_conf_samples) > 0:
            W2_modi[i] = current_cluster_high_conf_samples.mean(dim=0)
        else:
            W2_modi[i] = W2_initial[i] # 兜底：如果筛选后无样本，则使用原始中心
            
    W1_modi = orth_train(W1_modi, 512, use_relu=True)
    W2_modi = orth_train(W2_modi, cfg['backbone']['nclusters'], use_relu=True)

    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    torch.nn.init.zeros_(model.module.classify_tail[0].bias)
    torch.nn.init.zeros_(model.module.classify_tail[3].bias)
    torch.nn.init.zeros_(model.module.classify_medium[0].bias)
    torch.nn.init.zeros_(model.module.classify_medium[3].bias)
    
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()
    model.module.classify_tail[0].weight.data = W1_modi.clone()
    model.module.classify_tail[3].weight.data = W2_modi.clone()
    model.module.classify_medium[0].weight.data = W1_modi.clone()
    model.module.classify_medium[3].weight.data = W2_modi.clone()

    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()

# 分层聚类
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

def hierarchical_merge(W1, target_clusters):
    """
    使用层次聚类将 512 个聚类中心合并为 target_clusters 个
    """
    # [512, 512] => numpy
    W1_np = W1.detach().cpu().numpy()

    # 计算 pairwise 距离（默认欧氏）
    dist = pdist(W1_np, metric='euclidean')

    # 使用ward方法生成层次聚类树
    linkage_matrix = linkage(dist, method='ward')

    # 剪枝出 target_clusters 个簇
    labels = fcluster(linkage_matrix, target_clusters, criterion='maxclust')

    # 聚合为新中心
    new_centers = []
    for i in range(1, target_clusters + 1):
        idx = torch.tensor(labels == i)
        center = W1[idx].mean(dim=0)
        new_centers.append(center)

    return torch.stack(new_centers, dim=0)  # [K, D]

def initialize_weights_v5(cfg, model, cali_mlp, features, val_dataloader):
    features_zscore = (features - features.mean(1).reshape(-1, 1)) / features.std(1).reshape(-1, 1)
    features_zscore = F.normalize(features_zscore, dim=1)

    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_zscore)
    W1 = KMeans_512.cluster_centers_
    
    W1_orth = torch.empty_like(W1)
    torch.nn.init.orthogonal_(W1_orth)
    W1_matched = match_clusters_hungarian(W1_orth, W1)
    W1= W1_matched.clone()

    #linear(512,512)
    H = torch.mm(features, W1.T)
    #BN
    H = model.module.cluster_head[0][1](H).detach().clone()
    #relu
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)

    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=0)
    class_label = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_
    
    #W2 = hierarchical_merge(W1, cfg['backbone']['nclusters'])

    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, cfg['backbone']['nclusters'], use_relu=True)

    O = torch.mm(torch.mm(features, W1.T), W2.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    O = torch.mm(torch.mm(features, W1_modi.T) , W2_modi.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))

    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()

    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()

from scipy.optimize import linear_sum_assignment

def match_clusters_hungarian(W_orthogonal: torch.Tensor, W_kmeans: torch.Tensor) -> torch.Tensor:
    """
    使用匈牙利算法将正交初始化的权重 W_orthogonal 与 KMeans 聚类中心 W_kmeans 进行最优一一匹配。

    参数：
    - W_orthogonal: shape (n_clusters, feature_dim)，正交初始化矩阵
    - W_kmeans: shape (n_clusters, feature_dim)，KMeans 聚类中心

    返回：
    - W_matched: shape (n_clusters, feature_dim)，按匹配顺序排列的 W_orthogonal
    """

    # 确保维度一致
    assert W_orthogonal.shape == W_kmeans.shape
    n_clusters = W_orthogonal.shape[0]

    # 归一化向量（在余弦空间中进行匹配）
    W_orth = F.normalize(W_orthogonal, dim=1)  # (n, d)
    W_km = F.normalize(W_kmeans, dim=1)        # (n, d)

    # 计算余弦相似度矩阵（越大越相似）
    sim_matrix = torch.matmul(W_km, W_orth.T).cpu().numpy()  # (n, n)

    # 匈牙利算法要求是“成本矩阵”，所以取相反数（负相似度 = 正成本）
    cost_matrix = -sim_matrix

    # 求解最小代价匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 按 col_ind 索引 W_orth 中的行（即重排序）
    W_matched = W_orth[col_ind]

    return W_matched


# 多次聚类，取稳定样本
from scipy.stats import mode

def match_to_reference(ref_preds, preds, num_classes=None):
    if num_classes is None:
        num_classes = max(ref_preds.max(), preds.max()) + 1

    cost_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            cost_matrix[i][j] = -np.sum((ref_preds == i) & (preds == j))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {j: i for i, j in zip(row_ind, col_ind)}  # 使 preds 映射到 ref_preds 空间
    return np.array([mapping.get(p, -1) for p in preds])  # 加入默认值以防 KeyError

def get_stable_samples_by_matched_votes(kmeans_results, num_classes=10, vote_threshold=2):
    ref = kmeans_results[0]
    matched_preds = [ref]
    for i in range(1, len(kmeans_results)):
        aligned = match_to_reference(ref, kmeans_results[i], num_classes)
        matched_preds.append(aligned)
    
    matched_preds = np.stack(matched_preds, axis=1)  # [N, num_votes]
    modes, counts = mode(matched_preds, axis=1)
    stable_indices = np.where(counts[:, 0] >= vote_threshold)[0]
    
    #import pdb; pdb.set_trace()
    
    return stable_indices, modes[:, 0]

def initialize_weights_v6(cfg, model, cali_mlp, features, val_dataloader):
    features_zscore = (features - features.mean(1).reshape(-1, 1)) / features.std(1).reshape(-1, 1)
    features_zscore = F.normalize(features_zscore, dim=1)

    # 多次KMeans + 匈牙利匹配
    all_labels = []
    for seed in [0, 1, 2, 3, 4]:
        kmeans = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=seed)
        labels = kmeans.fit_predict(features_zscore)
        all_labels.append(labels.cpu().numpy())

    all_labels = np.stack(all_labels, axis=0)  # [n_runs, n_samples]
    
    ref_labels = all_labels[0]
    matched_labels = []
    for i in range(len(all_labels)):
        matched = match_to_reference(ref_labels, all_labels[i])
        matched_labels.append(matched)
    
    matched_labels = np.stack(matched_labels, axis=0)  # [n_runs, n_samples]

    # 投票选出一致性样本
    stable_indices, _ = get_stable_samples_by_matched_votes(matched_labels, num_classes=cfg['backbone']['nclusters'], vote_threshold=5)
    
    print("Stable indices:", len(stable_indices))

    # 用稳定样本重新聚类生成W1
    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    _ = KMeans_512.fit_predict(features_zscore[stable_indices])
    W1 = KMeans_512.cluster_centers_  # shape: [512, feat_dim]

    # 层2: 通过W1生成H再聚类得到W2
    H = torch.mm(features, W1.T)
    H = model.module.cluster_head[0][1](H).detach().clone()  # BN
    H = model.module.cluster_head[0][2](H).detach().clone()  # ReLU

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=0)
    _ = KMeans_c.fit_predict(H_zscore[stable_indices])
    W2 = KMeans_c.cluster_centers_

    # 正交化
    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, cfg['backbone']['nclusters'], use_relu=True)

    # 初始化 cluster_head
    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()

    # 初始化校准头
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()

    # 输出类别分布情况
    O = torch.mm(torch.mm(features, W1_modi.T), W2_modi.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    
    
# kmeans 512 时软分配特征 
def initialize_weights_v6_1(cfg, model, cali_mlp, features, val_dataloader): 
    features_zscore = (features - features.mean(1, keepdim=True)) / features.std(1, keepdim=True)
    features_zscore = F.normalize(features_zscore, dim=1)

    n_clusters = 512
    num_runs = 5
    all_labels = []

    # 多次 KMeans 聚类
    for seed in range(num_runs):
        kmeans = PyTorchKMeans(init='k-means++', n_clusters=n_clusters, verbose=False, random_state=seed)
        labels = kmeans.fit_predict(features_zscore)
        all_labels.append(labels.cpu().numpy())

    all_labels = np.stack(all_labels, axis=0)  # [num_runs, N]

    # 匹配聚类标签到参考标签
    ref_labels = all_labels[0]
    matched_labels = [match_to_reference(ref_labels, labels) for labels in all_labels]
    matched_labels = np.stack(matched_labels, axis=0)  # [num_runs, N]

    N = matched_labels.shape[1]
    cluster_dim = features.size(1)
    
    # 构建软分配矩阵 [N, n_clusters]
    soft_assignments = torch.zeros(N, n_clusters).to(features.device)  # 每个样本对每个类的概率

    for i in range(N):
        counts = Counter(matched_labels[:, i])
        for cls_id, count in counts.items():
            soft_assignments[i, cls_id] = count / num_runs

    # 计算加权中心 W1
    W1 = torch.zeros(n_clusters, cluster_dim).to(features.device)
    cluster_weights = torch.zeros(n_clusters).to(features.device)

    for i in range(N):
        for cls_id in range(n_clusters):
            weight = soft_assignments[i, cls_id]
            if weight > 0:
                W1[cls_id] += weight * features[i]
                cluster_weights[cls_id] += weight

    # 避免除以0
    cluster_weights[cluster_weights == 0] = 1e-6
    W1 /= cluster_weights.unsqueeze(1)

    # 计算H并进行第二层聚类（与之前一样）
    H = torch.mm(features, W1.T)
    H = model.module.cluster_head[0][1](H).detach().clone()  # BN
    H = model.module.cluster_head[0][2](H).detach().clone()  # ReLU

    H_zscore = (H - H.mean(1, keepdim=True)) / H.std(1, keepdim=True)
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=0)
    _ = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_

    # 正交化
    W1_modi = orth_train(W1, n_clusters, use_relu=True)
    W2_modi = orth_train(W2, cfg['backbone']['nclusters'], use_relu=True)

    # 初始化 cluster_head
    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()

    # 初始化校准头
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()

    # 输出类别分布情况
    O = torch.mm(torch.mm(features, W1_modi.T), W2_modi.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    
#全样本初始化w1，软分配初始化w2
def initialize_weights_v6_2(cfg, model, cali_mlp, features, val_dataloader):
    # 特征标准化
    features_zscore = (features - features.mean(1).reshape(-1, 1)) / features.std(1).reshape(-1, 1)
    features_zscore = F.normalize(features_zscore, dim=1)

    # 多次KMeans + 匈牙利匹配
    """ all_labels = []
    for seed in [0, 1, 2, 3, 4]:
        kmeans = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=seed)
        labels = kmeans.fit_predict(features_zscore)
        all_labels.append(labels.cpu().numpy())

    all_labels = np.stack(all_labels, axis=0)  # [n_runs, n_samples]
    ref_labels = all_labels[0]
    matched_labels = []
    for i in range(len(all_labels)):
        matched = match_to_reference(ref_labels, all_labels[i])
        matched_labels.append(matched)
    matched_labels = np.stack(matched_labels, axis=0)  # [n_runs, n_samples]

    # 获取每个样本的稳定性情况（投票频率）
    vote_counts = []
    for i in range(matched_labels.shape[1]):
        votes = matched_labels[:, i]
        counts = np.bincount(votes, minlength=512)
        vote_counts.append(counts / counts.sum())
    vote_distributions = np.stack(vote_counts, axis=0)  # [n_samples, 512] """

    # W1初始化：使用全部样本标准KMeans
    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    _ = KMeans_512.fit_predict(features_zscore)
    W1 = KMeans_512.cluster_centers_  # shape: [512, feat_dim]

    # 层2: 通过W1生成H再聚类得到W2
    H = torch.mm(features, W1.T)
    H = model.module.cluster_head[0][1](H).detach().clone()  # BN
    H = model.module.cluster_head[0][2](H).detach().clone()  # ReLU

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    H_zscore = F.normalize(H_zscore, dim=1)

    # W2初始化：使用软分配方式
    nclusters = cfg['backbone']['nclusters']
    class_feature_sum = torch.zeros(nclusters, H_zscore.shape[1]).cuda()
    class_weights = torch.zeros(nclusters).cuda()

    # 再聚类获取每次投票的H层标签
    all_H_labels = []
    for seed in [0, 1, 2, 3, 4]:
        kmeans_H = PyTorchKMeans(init='k-means++', n_clusters=nclusters, verbose=False, random_state=seed)
        labels_H = kmeans_H.fit_predict(H_zscore)
        all_H_labels.append(labels_H.cpu().numpy())

    all_H_labels = np.stack(all_H_labels, axis=0)  # [n_runs, n_samples]
    ref_labels_H = all_H_labels[0]
    matched_labels_H = []
    for i in range(len(all_H_labels)):
        matched_H = match_to_reference(ref_labels_H, all_H_labels[i])
        matched_labels_H.append(matched_H)
    matched_labels_H = np.stack(matched_labels_H, axis=0)
    
    # 投票选出一致性样本
    stable_indices, _ = get_stable_samples_by_matched_votes(matched_labels_H, num_classes=cfg['backbone']['nclusters'], vote_threshold=5)
    print(len(stable_indices), "stable samples of W2")

    # 构建软分配的方式（基于频率投票）
    print("Calculating soft assignments for W2...")
    for i in range(H_zscore.shape[0]):
        votes = matched_labels_H[:, i]
        counts = np.bincount(votes, minlength=nclusters)
        norm_counts = counts / counts.sum()
        for c in range(nclusters):
            weight = norm_counts[c]
            if weight > 0:
                class_feature_sum[c] += weight * H_zscore[i]
                class_weights[c] += weight

    # 求平均得到 W2
    W2 = class_feature_sum / (class_weights.view(-1, 1) + 1e-6)

    # 正交化
    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, nclusters, use_relu=True)

    # 初始化 cluster_head
    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()

    # 初始化校准头
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()

    # 输出类别分布情况
    O = torch.mm(torch.mm(features, W1_modi.T), W2_modi.T)
    print("Final output class distribution:", F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    
# v7
def select_center_and_hard_samples(features, pseudo_labels, n_clusters, center_ratio=0.5, hard_ratio=0.3):
    """
    分层选择中心样本和难样本用于初始化。

    Args:
        features (Tensor): 特征矩阵，形状为 [N, D]。
        pseudo_labels (ndarray): 聚类后的伪标签，形状为 [N]。
        n_clusters (int): 类别数量。
        center_ratio (float): 每个类别中，作为中心样本的比例。
        hard_ratio (float): 每个类别中，作为难样本的比例。

    Returns:
        selected_indices (List[int]): 被选中的样本索引列表。
    """
    features = features.cpu().numpy()
    selected_indices = []

    for c in range(n_clusters):
        cls_indices = np.where(pseudo_labels == c)[0]
        if len(cls_indices) < 5:
            continue

        cls_features = features[cls_indices]  # 当前类的所有特征
        center = cls_features.mean(axis=0, keepdims=True)  # 计算该类的中心

        dists = pairwise_distances(cls_features, center).reshape(-1)
        sorted_idx = np.argsort(dists)

        n_center = int(len(cls_indices) * center_ratio)
        n_hard = int(len(cls_indices) * hard_ratio)

        # 中心附近样本：最靠近中心的
        center_ids = cls_indices[sorted_idx[:n_center]]
        # 难样本：最远离中心的
        hard_ids = cls_indices[sorted_idx[-n_hard:]]

        selected_indices.extend(center_ids.tolist())
        selected_indices.extend(hard_ids.tolist())

    return selected_indices

def initialize_weights_v7(cfg, model, cali_mlp, features):

    features_zscore = (features - features.mean(1).reshape(-1, 1)) / features.std(1).reshape(-1, 1)
    features_zscore = F.normalize(features_zscore, dim=1)

    # 初始KMeans用于生成伪标签
    n_clusters = cfg['backbone']['nclusters']
    kmeans = PyTorchKMeans(init='k-means++', n_clusters=n_clusters, verbose=False, random_state=0)
    pseudo_labels = kmeans.fit_predict(features_zscore).cpu().numpy()

    # 选择中心样本和难样本
    selected_indices = select_center_and_hard_samples(features_zscore, pseudo_labels, n_clusters)

    print("Selected samples for init:", len(selected_indices))

    # 层1: 用选中样本重新聚512类
    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    _ = KMeans_512.fit_predict(features_zscore[selected_indices])
    W1 = KMeans_512.cluster_centers_

    # 层2: W1 → H → 再聚类
    H = torch.mm(features, W1.T)
    H = model.module.cluster_head[0][1](H).detach().clone()
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=n_clusters, verbose=False, random_state=0)
    _ = KMeans_c.fit_predict(H_zscore[selected_indices])
    W2 = KMeans_c.cluster_centers_

    # 正交化
    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, n_clusters, use_relu=True)

    # 初始化 cluster_head
    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()

    # 初始化 calibration_head
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()

    # 打印类别分布
    O = torch.mm(torch.mm(features, W1_modi.T), W2_modi.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    
    
import faiss
import matplotlib.pyplot as plt
import os 
# feature 稀疏化
def compute_feature_sparsity(features: np.ndarray, targets: np.ndarray, k=10):
    """
    features: [N, D] 特征矩阵
    targets: [N] 样本标签
    k: 近邻数量
    """

    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy().astype('float32')

    # 构建 faiss 索引（可换成 HNSW，如 IndexHNSWFlat）
    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)

    # 搜索 k+1 个最近邻（包含自己）
    distances, _ = index.search(features, k + 1)  # [N, k+1]

    # 排除第一个（自身距离为0）
    avg_distance = distances[:, 1:].mean(axis=1)  # 稀疏程度

    return avg_distance

def visualize_sparsity(avg_distance, targets, num_classes, save_path):
    """
    每类画一个 boxplot 或者 violin plot
    """
    data_per_class = [[] for _ in range(num_classes)]
    for d, t in zip(avg_distance, targets):
        data_per_class[t.item()].append(d)

    plt.figure(figsize=(12, 6))
    plt.boxplot(data_per_class, labels=[f"Class {i}" for i in range(num_classes)])
    plt.ylabel("Avg Distance to Neighbors (Sparsity)")
    plt.title("Feature Space Sparsity per Class")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sparsity_visualization.png'))
    
    
# 类内DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_class_intra_structure(cfg, features, labels, eps=2.0, min_samples=50, tsne_perplexity=40):
    """
    features: [N, D] 特征数组
    labels:   [N]    标签数组
    eps:      DBSCAN 的半径
    min_samples: 密度最小样本数
    """
    unique_classes = np.unique(labels)
    
    for c in unique_classes:
        print(f"\n🔍 分析类别 {c}：")
        class_mask = labels == c
        class_features = features[class_mask]
        class_features = class_features.cpu().numpy()
        
        #k, score = estimate_best_k(class_features, k_range=(2, 10))
        #print(f"Class {c}: best_k = {k}, silhouette_score = {score:.4f}")
        
        # 聚类
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(class_features)
        cluster_labels = db.labels_  # -1 为噪声点
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        print(f"  ➤ 子簇数: {n_clusters}")
        print(f"  ➤ 噪声点数量: {n_noise} / {len(class_features)}")
        
        # 子簇样本统计
        if n_clusters > 0:
            cluster_sizes = np.bincount(cluster_labels[cluster_labels != -1])
            print(f"  ➤ 每个子簇样本数: {cluster_sizes}")
        
        # 可视化
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
        features_2d = tsne.fit_transform(class_features)

        plt.figure(figsize=(8, 6))
        palette = sns.color_palette("tab10", n_colors=n_clusters + 1)
        for k in np.unique(cluster_labels):
            cluster_mask = cluster_labels == k
            label = f'Cluster {k}' if k != -1 else 'Noise'
            plt.scatter(features_2d[cluster_mask, 0],
                        features_2d[cluster_mask, 1],
                        s=20,
                        label=label,
                        alpha=0.7,
                        color=palette[k % len(palette)] if k != -1 else 'gray')

        plt.title(f'类别 {c} 的 DBSCAN 子簇分布图')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg['cdc_dir'], f'class_{c}_dbscan_clusters.png'))
        
# intra kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def estimate_best_k(features, k_range=(2, 10)):
    best_k = k_range[0]
    best_score = -1
    #features = features.cpu().numpy()  # 注意转为 CPU + numpy

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
        labels = kmeans.labels_
        score = silhouette_score(features, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score

def kmeans_intra_structure(cfg, probs, features, labels, t_label, tsne_perplexity=40): 
    unique_classes = np.unique(labels)
    conf = probs.max(1)[0].cpu().numpy()
    
    for c in unique_classes:
        print(f"\n🔍 分析类别 {c}：")
        class_mask = labels == c
        class_features = features[class_mask]
        class_features_np = class_features.cpu().numpy()
        
        conf_cls = conf[class_mask]
        
        # 1️⃣ 估计最佳聚类数 k
        best_k, score = estimate_best_k(class_features_np, k_range=(2, 6))
        print(f"Class {c}: best_k = {best_k}, silhouette_score = {score:.4f}")
        
        # 2️⃣ 进行KMeans聚类
        kmeans = KMeans(n_clusters=best_k, random_state=0).fit(class_features_np)
        cluster_labels = kmeans.labels_
        
        center_feature = np.mean(class_features_np, axis=0)
        
        weights = conf_cls / (conf_cls.sum() + 1e-8)  # 避免除0
        center_feature_weighted = np.sum(class_features_np * weights[:, None], axis=0)
        
        class_features_np = np.vstack([class_features_np, center_feature])
        class_features_np = np.vstack([class_features_np, center_feature_weighted])
        
        

        # 3️⃣ t-SNE降维
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
        tsne_feats = tsne.fit_transform(class_features_np)
        
        center_tsne_feats = tsne_feats[-2]
        center_tsne_feats_weighted = tsne_feats[-1]
        print(f"Center TSNE: {center_tsne_feats}, Weighted Center TSNE: {center_tsne_feats_weighted}")
        tsne_feats = tsne_feats[:-2]  # 去掉中心点和加权中心点

        
        # 4️⃣ 绘图
        plt.figure(figsize=(8, 6))
        center_np = []
        for k in range(best_k):
            idxs = cluster_labels == k
            plt.scatter(tsne_feats[idxs, 0], tsne_feats[idxs, 1], label=f'Sub-cluster {k}', s=10)
            print(f"sub-class{k}: ", len(tsne_feats[idxs]), conf_cls[idxs].mean(), conf_cls[idxs].std())
            
        """ center_np = np.array(center_np)
        center_avg = np.mean(center_np, axis=0)
        plt.scatter(center_avg[0], center_avg[1], marker='*', s=200, c='red', edgecolors='white', linewidths=1.5)
        #import pdb; pdb.set_trace() """
        
        # 绘制每个簇中心（⭐）
        plt.scatter(center_tsne_feats[0], center_tsne_feats[1], marker='*', s=200, c='black', edgecolors='white', linewidths=1.5)
        plt.scatter(center_tsne_feats_weighted[0], center_tsne_feats_weighted[1], marker='*', s=200, c='red', edgecolors='white', linewidths=1.5)
        
        plt.title(f"t-SNE of Class {c} (Best k={best_k}, Silhouette={score:.4f})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg['cdc_dir'], f'class_{c}_kmeans_clusters.png'))
        
        
        # 5️⃣ 基于真实标签绘制 t-SNE 分布
        t_label_np = t_label[class_mask].cpu().numpy()
        tsne_feats_true = tsne_feats  # 与上面复用降维结果

        plt.figure(figsize=(8, 6))
        unique_true_classes = np.unique(t_label_np)
        for t in unique_true_classes:
            idxs = t_label_np == t
            plt.scatter(tsne_feats_true[idxs, 0], tsne_feats_true[idxs, 1], label=f'True class {t}', s=10)

        plt.title(f"t-SNE of Real Labels in Pseudo-Class {c}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(cfg['cdc_dir'], f'class_{c}_true_label_tsne.png'))
               
# 基于置信度加权类中心
def initialize_weights_v8(cfg, probs, model, cali_mlp, features):
    features_zscore = (features - features.mean(1).reshape(-1, 1)) / features.std(1).reshape(-1, 1)
    #features_zscore = features.detach()
    features_zscore = F.normalize(features_zscore, dim=1)

    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_zscore)
    W1 = KMeans_512.cluster_centers_

    #linear(512,512)
    H = torch.mm(features, W1.T)
    #BN
    H = model.module.cluster_head[0][1](H).detach().clone()
    #relu
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False, random_state=0)
    
    class_label = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_

    W2_weighted = []
    for i in range(cfg['backbone']['nclusters']):
        #mask = (class_label == i)
        mask = (class_label == i).to(probs.device)  # 保证 mask 和 probs 在同一个设备

        if mask.sum() == 0:
            # 若某个簇为空，保留原KMeans中心
            W2_weighted.append(W2[i])
            continue

        class_features_np = H_zscore[mask].cpu().numpy()      # [n_i, d]
        conf = probs[mask, i].detach().cpu().numpy()          # [n_i]
        conf = conf / (conf.sum() + 1e-6)                     # 归一化

        weighted_center = np.sum(class_features_np * conf[:, None], axis=0)
        W2_weighted.append(weighted_center)

    W2_weighted = torch.tensor(np.stack(W2_weighted), dtype=torch.float32).cuda()
    W2 = W2_weighted.clone()

    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, cfg['backbone']['nclusters'], use_relu=True)
    
    O = torch.mm(torch.mm(features, W1.T), W2.T)
    O = torch.mm(torch.mm(features, W1_modi.T) , W2_modi.T)

    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()
    
# v9   
def select_informative_samples(probs, features, top_k_ratio=0.1, bottom_k_ratio=0.1, per_class_keep_ratio=0.8):
    """
    针对每个类别，选择置信度适中且特征多样的样本，确保类别均衡。
    
    参数：
        probs: [N, C] softmax 概率
        features: [N, D] 特征
        top_k_ratio: 丢弃最 confident 的比例（easy）
        bottom_k_ratio: 丢弃最不 confident 的比例（hard）
        per_class_keep_ratio: 每类保留 informative 样本的比例（根据特征多样性选）
    
    返回：
        selected_final: tensor of selected indices，设备和 features 一致
    """
    with torch.no_grad():
        device = features.device
        confidences, pred_classes = torch.max(probs, dim=1)
        selected_indices = []

        num_classes = probs.size(1)

        for cls in range(num_classes):
            cls_mask = (pred_classes == cls)
            cls_indices = torch.nonzero(cls_mask, as_tuple=False).squeeze()
            if cls_indices.numel() == 0:
                continue

            cls_conf = confidences[cls_indices]
            cls_feats = features[cls_indices]

            sorted_conf_idx = torch.argsort(cls_conf)
            n = len(cls_conf)
            topk = int(n * top_k_ratio)
            bottomk = int(n * bottom_k_ratio)

            informative_idx = sorted_conf_idx[bottomk: n - topk]
            cls_feats_sel = cls_feats[informative_idx]

            # 多样性：选取 feature 分布中最分散的样本（越不像别人的越好）
            cosine_sim = torch.mm(F.normalize(cls_feats_sel, dim=1), F.normalize(cls_feats_sel, dim=1).T)
            sim_score = cosine_sim.mean(dim=1)
            diverse_idx = torch.argsort(-sim_score)  # 倒序：越“不同”排越前

            keep_num = int(len(diverse_idx) * per_class_keep_ratio)
            if keep_num == 0:
                continue

            final_idx = informative_idx[diverse_idx[:keep_num].to(informative_idx.device)]

            selected_indices.append(cls_indices[final_idx])

        if len(selected_indices) == 0:
            return torch.tensor([], dtype=torch.long, device=device)

        selected_final = torch.cat(selected_indices).to(device)
        return selected_final

def initialize_weights_v9(cfg, probs, model, cali_mlp, features, val_dataloader):
    selected_idx = select_informative_samples(probs, features)
    features = features[selected_idx]
    
    print(len(selected_idx), "informative samples selected for initialization")

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
    
    
#v10 longtail

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

    #pdb.set_trace()

    # import matplotlib.pyplot as plt
    # dis_norm = (dis - dis.min()) / (dis.max() - dis.min())
    # plt.hist(dis_norm, bins=50, density=True, alpha=0.6, color='g', label="Histogram")
    # from scipy.stats import gaussian_kde
    # kde = gaussian_kde(dis_norm)
    # x = np.linspace(0, 1, 200)
    # plt.plot(x, kde(x), 'r-', lw=2, label="KDE")
    # plt.title("Normalized dis Distribution")
    # plt.xlabel("dis (normalized)")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.savefig("num-dis1.png")
    # #pdb.set_trace()

    # # 用 numpy.histogram 统计
    # counts, bin_edges = np.histogram(dis_norm, bins=50)

    # # 打印每个区间的样本数
    # for i in range(len(counts)):
    #     print(f"区间 {bin_edges[i]:.2f} ~ {bin_edges[i+1]:.2f} : {counts[i]} 个样本")

    # 转回 torch，并放回原来的 device
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


def initialize_weights_bias(cfg, model, cali_mlp, features, val_dataloader, k=10, alpha=1.0, target_class= 3):
    # 特征预处理
    features_zscore = (features - features.mean(1, keepdim=True)) / features.std(1, keepdim=True)
    features_zscore = F.normalize(features_zscore, dim=1)

    # Step1: 先KMeans 512
    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False, random_state=0)
    proto_label = KMeans_512.fit_predict(features_zscore)
    proto_label = torch.tensor(proto_label, device=features.device)

    """ # Step2: 用密度加权重新计算 W1
    density_weights = compute_density_weights(features_zscore, proto_label, k=k, alpha=alpha)
    W1 = weighted_cluster_centers(features, proto_label, 512, density_weights) """
    W1 = KMeans_512.cluster_centers_

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
    density_weights2= compute_density_weights(H_zscore, class_label, k=k, alpha=alpha)
    density_weights2 = density_weights2 * (len(density_weights2) / (density_weights2.sum() + 1e-6))
    W2 = weighted_cluster_centers(H_zscore, class_label, cfg['backbone']['nclusters'], density_weights2)

    #W2 = KMeans_c.cluster_centers_

    # density_weights2_np = density_weights2.detach().cpu().numpy()
    # counts_low = 0
    # counts_mid = 0
    # counts_high = 0
    # for i in range(len(counts)):
    #     if i<17:
    #         counts_high+=counts[i]
    #     elif i<30:
    #         counts_mid+=counts[i]
    #     else:
    #         counts_low+=counts[i]

    
    # indices_per_bin = []
    # sorted_idx = np.argsort(-density_weights2_np)
    # high_idx = sorted_idx[:counts_high]
    # mid_idx  = sorted_idx[counts_high:counts_high+counts_mid]
    # low_idx  = sorted_idx[counts_high+counts_mid:counts_high+counts_mid+counts_low]
    # indices_per_bin = [low_idx, mid_idx, high_idx]

    #pdb.set_trace()
    #percentiles = np.percentile(density_weights2_np, [5,35,65,95])
    """ percentiles = [0.33, 0.66]
    indices_per_bin = []
    for i in range(5):
        if i == 0:
            mask = density_weights2_np <= percentiles[i]
        elif i == 4:
            mask = density_weights2_np > percentiles[i-1]
        else:
            mask = (density_weights2_np > percentiles[i-1]) & (density_weights2_np <= percentiles[i])

        # 进一步筛选出 target_class 的样本
        #class_mask = (class_label == target_class).cpu().numpy().astype(bool)
        #mask = mask & class_mask

        bin_indices = np.where(mask)[0]  # 该区间内目标类别的 index
        
        # 随机选取10个 index（如果不足10个就全取）
        #chosen = np.random.choice(bin_indices, size=min(10, len(bin_indices)), replace=False)
        indices_per_bin.append(bin_indices) """

    #print(indices_per_bin)
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