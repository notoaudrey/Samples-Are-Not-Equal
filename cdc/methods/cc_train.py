'''
@File  :cc_train.py
@Author:cjh
@Date  :2023/2/25 16:48
@Desc  :
'''
from cdc.losses.losses import CCInstanceLoss, CCClusterLoss
#from torch.cuda.amp import autocast, GradScaler
import torch
import torch.nn.functional as F
from collections import Counter


def cc_train(cfg, clustering_stats, train_dataloader, model, criterion, optimizer, epoch):
    model.train()
    for i, batch in enumerate(train_dataloader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        gt = batch['target'].cuda(non_blocking=True)

        if len(clustering_stats) != 0:
            gt_map = clustering_stats['hungarian_match']
        for pre, post in gt_map:
            gt[batch['target'] == post] = pre

        z_i, z_j, c_i, c_j = model(images, images_augmented)
        loss_instance = CCInstanceLoss(cfg['optimizer']['batch_size'], cfg['criterion']['instance_temperature'])(z_i, z_j)
        loss_cluster, ne_loss = CCClusterLoss(cfg['backbone']['nclusters'], cfg['criterion']['cluster_temperature'])(c_i, c_j)

        loss = loss_instance + loss_cluster + ne_loss * cfg['method_kwargs']['entropy_loss_weight']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
def divclust_cc_train(cfg, train_dataloader, model, criterion, optimizer, epoch):
    model.train()
    for i, batch in enumerate(train_dataloader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        gt = batch['target'].cuda(non_blocking=True)

        loss, metrics_dict = model(images, images_augmented, forward_pass='loss')
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   
        
        
def cc_train_longtail(cfg, clustering_stats, train_dataloader, model, criterion, optimizer, epoch, tail_neighbors_idx, medium_neighbors_idx, pseudo_labels):
    model.train()
    for i, batch in enumerate(train_dataloader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        gt = batch['target'].cuda(non_blocking=True)

        if len(clustering_stats) != 0:
            gt_map = clustering_stats['hungarian_match']
        for pre, post in gt_map:
            gt[batch['target'] == post] = pre

        z_i, z_j, c_i, c_j = model(images, images_augmented)
        loss_instance = CCInstanceLoss(cfg['optimizer']['batch_size'], cfg['criterion']['instance_temperature'])(z_i, z_j)
        loss_cluster, ne_loss = CCClusterLoss(cfg['backbone']['nclusters'], cfg['criterion']['cluster_temperature'])(c_i, c_j)
        
        # ========== Neighbor-Enhanced Label Smoothing ==========
        epsilon = cfg['epsilon']
        num_class = cfg['backbone']['nclusters']
        #### Step 1: 伪标签 one-hot
        batch_size = images.shape[0]
        #pseudo = torch.tensor(pseudo_labels)[indices].long().cuda()
        pseudo = torch.tensor(pseudo_labels)[images_index.cpu()].long().cuda()
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
            tail_neigh = tail_neighbors_idx[images_index[j].item()]
            medium_neigh = medium_neighbors_idx[images_index[j].item()]

            if len(tail_neigh) > 0:
                target_tail[j, tail_neigh] += epsilon / len(tail_neigh)
            if len(medium_neigh) > 0:
                target_medium[j, medium_neigh] += epsilon / len(medium_neigh)

        target_tail = target_tail / target_tail.sum(dim=1, keepdim=True)
        target_medium = target_medium / target_medium.sum(dim=1, keepdim=True)
        
        # 类频率平衡（spc）
        from collections import Counter
        spc_dict = Counter(pseudo_labels.cpu().numpy())  # {cluster_id: count}
        n_clusters = cfg['backbone']['nclusters']
        # 构建 spc 列表，并确保最小值为1（避免 log(0)）
        spc = [spc_dict.get(i, 1) for i in range(n_clusters)]
        # 转换为 Tensor 并放到 CUDA
        spc = torch.tensor(spc, dtype=torch.float32).cuda()

        # logits 调整 + loss
        logits_tail = c_i + 1.0 * spc.log()
        logits_medium = c_i + 0.5 * spc.log()

        loss_tail = -torch.sum(F.log_softmax(logits_tail, dim=1) * target_tail) / batch_size
        loss_medium = -torch.sum(F.log_softmax(logits_medium, dim=1) * target_medium) / batch_size

        loss = loss_instance + loss_cluster + ne_loss * cfg['method_kwargs']['entropy_loss_weight'] + loss_tail + loss_medium
        
        #import pdb; pdb.set_trace()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
