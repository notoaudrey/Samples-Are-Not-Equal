'''
@File  :scan_train.py
@Author:cjh
@Date  :2023/2/25 19:17
@Desc  :
'''
import torch

from cdc.losses.losses import SCANLoss, ConfidenceBasedCE
from cdc.utils.torch_clustering import PyTorchKMeans
import torch.nn.functional as F
from collections import Counter

from torch.utils.data import Subset, DataLoader

def freeze_backbone(model):
    """冻结 model.module.backbone 的所有参数"""
    for param in model.module.backbone.parameters():
        param.requires_grad = False
    print("[INFO] Backbone frozen.")



def select_high_confidence_samples(model, dataloader, top_ratio=0.1):
    """
    返回高置信度样本的 index 列表和其对应的伪标签。

    返回：
        top_sample_indices (Tensor): 高置信度样本的全局索引
        top_pseudo_labels (Tensor): 对应伪标签
    """
    model.eval()
    all_indices, all_confidences, all_pseudo_labels = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0]
            indices = batch[2]
            outputs = model(images.cuda(non_blocking=True),
                            forward_pass='return_all')['output'][0]
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)

            all_indices.append(indices)
            all_confidences.append(confs.cpu())
            all_pseudo_labels.append(preds.cpu())

    all_indices = torch.cat(all_indices)
    all_confidences = torch.cat(all_confidences)
    all_pseudo_labels = torch.cat(all_pseudo_labels)

    """ num_top = int(top_ratio * len(all_confidences))
    top_indices = torch.argsort(all_confidences, descending=True)[:num_top] """
    
    threshold = 0.3  # 或你想设定的其他值
    top_mask = all_confidences > threshold
    top_indices = torch.nonzero(top_mask, as_tuple=False).squeeze()
    print(f"[Confidence Filter] Selected {len(top_indices)} samples with confidence > {threshold}")

    if len(top_indices)==0:
        num_top = int(top_ratio * len(all_confidences))
        top_indices = torch.argsort(all_confidences, descending=True)[:num_top]

    top_sample_indices = all_indices[top_indices]
    top_pseudo_labels = all_pseudo_labels[top_indices]

    return top_sample_indices, top_pseudo_labels

def train_with_pseudo_labels(cfg, model, dataset, optimizer, sample_indices, pseudo_labels):
    """
    使用高置信度伪标签样本进行一次小批次训练（仅1轮）。

    参数：
    - dataset: 原始训练集（非 dataloader）
    - sample_indices: 高置信度样本索引（Tensor）
    - pseudo_labels: 与索引对应的标签（Tensor）
    """
    # 建立索引 -> 标签映射表
    label_map = {idx.item(): label.item() for idx, label in zip(sample_indices, pseudo_labels)}

    subset = Subset(dataset, sample_indices)

    def pseudo_label_collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        indices = [item['index'].item() for item in batch]
        targets = torch.tensor([label_map[idx] for idx in indices])
        return images, targets

    loader = DataLoader(subset,
                        batch_size=cfg['optimizer']['batch_size'],
                        shuffle=True,
                        num_workers=cfg.get('num_workers', 4),
                        pin_memory=True,
                        collate_fn=pseudo_label_collate_fn)

    model.train()
    
    freeze_backbone(model)

    total_loss = 0
    for images, targets in loader:
        images = images.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)[0]  # 主头输出
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    avg_loss = total_loss / len(loader.dataset)
    print(f"[Pseudo-label Training] Top {len(loader.dataset)} samples, avg loss: {avg_loss:.4f}")

def scan_train(cfg, clustering_stats, train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """
    Train w/ SCAN-Loss
    """
    model.train()  # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = batch['image'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)

        anchors_output = model(anchors)
        neighbors_output = model(neighbors)

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = SCANLoss(cfg['method_kwargs']['entropy_weight'])(anchors_output_subhead,
                                                                      neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        total_loss = torch.sum(torch.stack(total_loss, dim=0))
        
        #import pdb; pdb.set_trace()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

def selflabel_train(cfg, clustering_stats, train_loader, model, criterion, optimizer, epoch, ema=None):
    """
        Self-labeling based on confident samples
        """
    model.train()
    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        gt = batch['target'].cuda(non_blocking=True)

        if len(clustering_stats) != 0:
            gt_map = clustering_stats['hungarian_match']
            for pre, post in gt_map:
                gt[batch['target'] == post] = pre

        with torch.no_grad():
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]

        loss = ConfidenceBasedCE(cfg['method_kwargs']['threshold'],
                                cfg['method_kwargs']['apply_class_balancing'])(output, output_augmented)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def scan_train_longtail(cfg, train_loader, model, optimizer, epoch,
                        tail_neighbors_idx, medium_neighbors_idx, pseudo_labels):
    """
    SCAN training with long-tail aware expert branches and soft neighbor-enhanced targets.
    """
    model.train()

    epsilon = cfg['epsilon']
    entropy_weight = cfg['method_kwargs']['entropy_weight']
    num_class = cfg['backbone']['nclusters']

    for i, batch in enumerate(train_loader):
        anchors = batch['image'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
        indices = batch['index'].cuda()
        
        anchors_res = model(anchors, forward_pass='return_all')
        anchors_feats, anchors_outputs = anchors_res['features'], anchors_res['output']
        neighbors_res = model(neighbors, forward_pass='return_all')
        neighbors_outputs = neighbors_res['output']

        total_loss = 0
        for out_a, out_n in zip(anchors_outputs, neighbors_outputs):
            scan_loss, cons_loss, ent_loss = SCANLoss(entropy_weight)(out_a, out_n)
            total_loss += scan_loss

        #### Step 1: 伪标签 one-hot
        batch_size = anchors.shape[0]
        #pseudo = torch.tensor(pseudo_labels)[indices].long().cuda()
        pseudo = torch.tensor(pseudo_labels)[indices.cpu()].long().cuda()

        pseudo_onehot = torch.zeros(batch_size, num_class).cuda()
        pseudo_onehot.scatter_(1, pseudo.view(-1,1), 1)

        #### Step 2: 构建 soft targets for tail / medium expert
        target_tail = torch.zeros_like(pseudo_onehot)
        target_medium = torch.zeros_like(pseudo_onehot)

        for j in range(batch_size):
            label = pseudo[j].item()
            # Base confidence
            target_tail[j, label] = 1 - epsilon
            target_medium[j, label] = 1 - epsilon

            # neighbor-enhanced confidence
            tail_neigh = tail_neighbors_idx[indices[j].item()]
            medium_neigh = medium_neighbors_idx[indices[j].item()]

            if len(tail_neigh) > 0:
                target_tail[j, tail_neigh] += epsilon / len(tail_neigh)
            if len(medium_neigh) > 0:
                target_medium[j, medium_neigh] += epsilon / len(medium_neigh)

        target_tail = target_tail / target_tail.sum(dim=1, keepdim=True)
        target_medium = target_medium / target_medium.sum(dim=1, keepdim=True)

        #### Step 3: 从专家 head 获取输出
        logits_tail = model.module.classify_tail(anchors_feats)
        logits_medium = model.module.classify_medium(anchors_feats)

        spc_dict = Counter(pseudo_labels.cpu().numpy())  # {cluster_id: count}
        n_clusters = cfg['backbone']['nclusters']

        # 构建 spc 列表，并确保最小值为1（避免 log(0)）
        spc = [spc_dict.get(i, 1) for i in range(n_clusters)]

        # 转换为 Tensor 并放到 CUDA
        spc = torch.tensor(spc, dtype=torch.float32).cuda()
        
        """ adj_tail = logits_tail + 1.0 * spc.log()
        adj_medium = logits_medium + 0.5 * spc.log() """
        adj_tail = logits_tail 
        adj_medium = logits_medium 

        loss_tail = -torch.sum(F.log_softmax(adj_tail, dim=1) * target_tail) / batch_size
        loss_medium = -torch.sum(F.log_softmax(adj_medium, dim=1) * target_medium) / batch_size

        #### Step 4: 合并损失
        loss = total_loss + loss_tail + loss_medium

        #import pdb; pdb.set_trace()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
def scan_train_LMv1(cfg, train_loader, model, optimizer, epoch,
                        tail_neighbors_idx, medium_neighbors_idx, pseudo_labels, ratio=1):
    """
    SCAN training with label smoothing.
    """
    model.train()

    epsilon = cfg['epsilon']
    entropy_weight = cfg['method_kwargs']['entropy_weight']
    num_class = cfg['backbone']['nclusters']
    
    consistency_indices = []

    for i, batch in enumerate(train_loader):
        anchors = batch['image'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
        indices = batch['index'].cuda()
        
        anchors_res = model(anchors, forward_pass='return_all')
        anchors_feats, anchors_outputs = anchors_res['features'], anchors_res['output']
        anchors_feats.requires_grad_(True)
        anchors_feats.retain_grad() 
        
        neighbors_res = model(neighbors, forward_pass='return_all')
        neighbors_outputs = neighbors_res['output']

        total_loss = 0
        for out_a, out_n in zip(anchors_outputs, neighbors_outputs):
            scan_loss, cons_loss, ent_loss = SCANLoss(entropy_weight)(out_a, out_n)
            total_loss += scan_loss


        #### 伪标签 one-hot
        batch_size = anchors.shape[0]
        pseudo = torch.tensor(pseudo_labels)[indices.cpu()].long().cuda()
        pseudo_onehot = torch.zeros(batch_size, num_class).cuda()
        pseudo_onehot.scatter_(1, pseudo.view(-1,1), 1)

        target_tail = torch.zeros_like(pseudo_onehot)

        for j in range(batch_size):
            label = pseudo[j].item()
            target_tail[j, label] = 1 - epsilon
            # neighbor-enhanced confidence
            tail_neigh = tail_neighbors_idx[indices[j].item()]

            if len(tail_neigh) > 0:
                target_tail[j, tail_neigh] += epsilon / len(tail_neigh)

        target_tail = target_tail / target_tail.sum(dim=1, keepdim=True)

        logits_tail = model.module.classify_tail(anchors_feats)
        logits_tail.requires_grad_(True)
        logits_tail.retain_grad()

        spc_dict = Counter(pseudo_labels.cpu().numpy())  # {cluster_id: count}
        n_clusters = cfg['backbone']['nclusters']
        spc = [spc_dict.get(i, 1) for i in range(n_clusters)]
        spc = torch.tensor(spc, dtype=torch.float32).cuda()
        adj_tail = logits_tail + 1.0 * spc.log()

        loss_tail = -torch.sum(F.log_softmax(adj_tail, dim=1) * target_tail) / batch_size

        #### Step 4: 合并损失
        loss = total_loss + ratio * loss_tail

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
def scan_train_LMv2(cfg, train_loader, model, optimizer, epoch,
                        tail_neighbors_idx, medium_neighbors_idx, pseudo_labels):
    """
    SCAN training with label smoothing.
    """
    model.train()

    epsilon = cfg['epsilon']
    entropy_weight = cfg['method_kwargs']['entropy_weight']
    num_class = cfg['backbone']['nclusters']

    for i, batch in enumerate(train_loader):
        anchors = batch['image'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
        indices = batch['index'].cuda()
        
        anchors_res = model(anchors, forward_pass='return_all')
        anchors_feats, anchors_outputs = anchors_res['features'], anchors_res['output']
        neighbors_res = model(neighbors, forward_pass='return_all')
        neighbors_outputs = neighbors_res['output']

        total_loss = 0
        for out_a, out_n in zip(anchors_outputs, neighbors_outputs):
            scan_loss, cons_loss, ent_loss = SCANLoss(entropy_weight)(out_a, out_n)
            total_loss += scan_loss


        #### 伪标签 one-hot
        batch_size = anchors.shape[0]
        pseudo = torch.tensor(pseudo_labels)[indices.cpu()].long().cuda()
        pseudo_onehot = torch.zeros(batch_size, num_class).cuda()
        pseudo_onehot.scatter_(1, pseudo.view(-1,1), 1)

        target_tail = torch.zeros_like(pseudo_onehot)

        for j in range(batch_size):
            label = pseudo[j].item()
            target_tail[j, label] = 1 - epsilon
            # neighbor-enhanced confidence
            tail_neigh = tail_neighbors_idx[indices[j].item()]

            if len(tail_neigh) > 0:
                target_tail[j, tail_neigh] += epsilon / len(tail_neigh)

        target_tail = target_tail / target_tail.sum(dim=1, keepdim=True)

        logits_tail = model.module.classify_tail(anchors_feats)

        spc_dict = Counter(pseudo_labels.cpu().numpy())  # {cluster_id: count}
        n_clusters = cfg['backbone']['nclusters']
        spc = [spc_dict.get(i, 1) for i in range(n_clusters)]
        spc = torch.tensor(spc, dtype=torch.float32).cuda()
        adj_tail = logits_tail + 1.0 * spc.log()

        loss_tail = -torch.sum(F.log_softmax(adj_tail, dim=1) * target_tail) / batch_size
        
        loss = loss_tail

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
              
def scan_train_LMv3(cfg, train_loader, model, optimizer, epoch,
                        tail_neighbors_idx, medium_neighbors_idx, pseudo_labels):
    """
    SCAN training with label smoothing.
    """
    model.train()

    epsilon = cfg['epsilon']
    entropy_weight = cfg['method_kwargs']['entropy_weight']
    num_class = cfg['backbone']['nclusters']

    for i, batch in enumerate(train_loader):
        anchors = batch['image'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)
        indices = batch['index'].cuda()
        
        anchors_res = model(anchors, forward_pass='return_all')
        anchors_feats, anchors_outputs = anchors_res['features'], anchors_res['output']
        neighbors_res = model(neighbors, forward_pass='return_all')
        neighbors_outputs = neighbors_res['output']

        total_loss = 0
        for out_a, out_n in zip(anchors_outputs, neighbors_outputs):
            scan_loss, cons_loss, ent_loss = SCANLoss(entropy_weight)(out_a, out_n)
            total_loss += scan_loss


        #### 伪标签 one-hot
        batch_size = anchors.shape[0]
        pseudo = torch.tensor(pseudo_labels)[indices.cpu()].long().cuda()
        pseudo_onehot = torch.zeros(batch_size, num_class).cuda()
        pseudo_onehot.scatter_(1, pseudo.view(-1,1), 1)

        target_tail = torch.zeros_like(pseudo_onehot)

        for j in range(batch_size):
            label = pseudo[j].item()
            target_tail[j, label] = 1

        target_tail = target_tail / target_tail.sum(dim=1, keepdim=True)

        logits_tail = model.module.classify_tail(anchors_feats)

        adj_tail = logits_tail

        loss_tail = -torch.sum(F.log_softmax(adj_tail, dim=1) * target_tail) / batch_size
        
        loss = loss_tail + total_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
def init_head_singlelayer(model, features, predictions, n_clusters, 
                          top_ratio=0.5, confidence_offset=0.0, 
                          balanced_per_class=False):
    """
    用高置信度样本做 KMeans 初始化单层分类头
    参数:
        model: 模型对象，包含 cluster_head[0] (nn.Linear)
        features: Tensor[N, D]，所有样本特征
        predictions: Tensor[N, C]，每个样本的 softmax logits
        n_clusters: 聚类类别数
        top_ratio: 每类或全局选择的比例
        confidence_offset: 置信度偏移（跳过最前面的一部分高置信度样本）
        balanced_per_class: 是否每类均衡采样

    返回:
        None（直接修改 model.cluster_head[0]）
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
    print("ini sample:", len(top_indices))

    # 归一化特征
    features_zscore = (features_sub - features_sub.mean(1, keepdim=True)) / (features_sub.std(1, keepdim=True) + 1e-6)
    features_zscore = F.normalize(features_zscore, dim=1)

    # KMeans 聚类
    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=n_clusters, verbose=False, random_state=0)
    _ = KMeans_c.fit_predict(features_zscore)
    W1 = KMeans_c.cluster_centers_

    # 初始化分类头
    with torch.no_grad():
        torch.nn.init.zeros_(model.module.cluster_head[0].bias)
        model.module.cluster_head[0].weight.data = W1.clone()

    print("cluster_head[0] initialized with shape:", W1.shape)


        
        