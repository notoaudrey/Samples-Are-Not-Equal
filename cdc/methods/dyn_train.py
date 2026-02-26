import torch
from collections import defaultdict, deque
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import Sampler
import random


class DynamicSampler(Sampler):
    def __init__(self, tracker, shuffle=True):
        self.tracker = tracker
        self.num_samples = tracker.num_samples
        self.shuffle = shuffle

    def __iter__(self):
        available_indices = [i for i in range(self.num_samples) if i not in self.tracker.removed]
        if self.shuffle:
            random.shuffle(available_indices)
        return iter(available_indices)

    def __len__(self):
        return self.num_samples - len(self.tracker.removed)

class SampleMasterTracker:
    def __init__(self, cfg, num_samples, delta_thresh=1e-3, window=3, min_cluster_ratio=0.2, shake_thresh=1.0, shake_epoch=-1, s=0.2):
        """
        Args:
            num_samples (int): 样本总数
            delta_thresh (float): 二阶差分阈值（稳定）
            window (int): 历史窗口大小（推荐3）
            min_cluster_ratio (float): 每个簇至少保留比例，防止全被移除
            shake_thresh (float): 二阶差分阈值（抖动）
        """
        self.num_samples = num_samples
        self.delta_thresh = delta_thresh
        self.shake_thresh = shake_thresh
        self.shake_epoch = shake_epoch
        self.window = window
        self.min_cluster_ratio = min_cluster_ratio

        self.conf_history = defaultdict(lambda: deque(maxlen=window))
        self.label_history = defaultdict(lambda: deque(maxlen=window))
        self.loss_history = defaultdict(lambda: deque(maxlen=window))

        self.removed = set()  # 已移除（不回传梯度）的样本索引
        self.shake = set()    # 抖动的样本索引
        # self.highconf =set()
        self.shake_indices = set()

        self.restore_log = [] # 恢复的历史
        
        self.delta2_history = []  # 每次step保存全体样本delta2
        self.shake_delta2_history = [] 
        self.cfg = cfg
        self.log_path = os.path.join(cfg['cdc_dir'], 'training_log.log')

        self.beta, self.mu = 0.5, 1.0
        self.s=s
        self.epoch = 0

    def get_sampler(self):
        return DynamicSampler(self.num_samples, self.removed)


    def update(self, indices, labels, losses):
        """
        更新样本的置信度和伪标签历史
        Args:
            indices (list[int]): 样本索引
            confidences (list[float]): 样本对应的置信度
            labels (list[int]): 样本对应的伪标签
        """
        for idx, lab, loss in zip(indices, labels, losses):
            #self.conf_history[idx].append(conf)
            self.label_history[idx].append(lab)
            self.loss_history[idx].append(loss)

    def step(self):
        """
        在一个 epoch 或一个大 step 结束后调用，更新 removed 集合
        """
        new_removed, new_restore = set(), set()
        new_shake, restore_shake = set(), set()
        delta2_all = [] 
        loss_all= []
        self.epoch+=1

        for idx in range(self.num_samples):
            if len(self.loss_history[idx]) < self.window:
                continue
            # 计算二阶差分
            c = self.loss_history[idx]
            delta2 = c[-1] - 2 * c[-2] + c[-3]
            delta2_all.append(abs(delta2))
            stable_label = len(set(self.label_history[idx])) == 1
            #high_conf = min(set(self.conf_history[idx])) > 0.9
            mean_loss = sum(c) / len(c)
            loss_all.append(mean_loss)
            #high_stability = mean_loss < self.s
            # high_conf = sum(self.conf_history[idx])/len(self.conf_history[idx]) > 0.9
            if abs(delta2) < self.delta_thresh and stable_label:
                new_removed.add(idx)

            if abs(delta2) > self.shake_thresh:
                new_shake.add(idx)

            """ if stable_label and high_conf:
                self.highconf.add(idx) """

            # 如果之前被移除，但现在不满足条件 → 恢复
            if idx in self.removed and (abs(delta2) >= self.delta_thresh or not stable_label):
                new_restore.add(idx)

            if idx in self.shake and (abs(delta2) <= self.shake_thresh):
                restore_shake.add(idx)

            """ if idx in self.highconf and (not high_conf or not stable_label):
                self.highconf.remove(idx) """

        #pdb.set_trace()

        # 更新 removed 集合
        self.removed = (self.removed | new_removed) - new_restore
        self.shake = (self.shake | new_shake) - restore_shake

        """ if self.epoch == 10:
            self.shake_indices = self.shake """

        # 记录恢复日志
        if len(new_restore) > 0:
            self.restore_log.append((len(self.restore_log), list(new_restore)))

        """ # 采用random策略
        num_to_remove = int(0.5 * self.num_samples)
        random_removed = set(random.sample(range(self.num_samples), num_to_remove))
        self.removed = random_removed """

        # 🔥 输出当前可参与反向传播的样本数
        num_active = self.num_samples - len(self.removed)
        print(f"Active samples for backprop: {num_active}/{self.num_samples} (removed {len(self.removed)})")
        print(f"samples shake: {len(self.shake)}/{self.num_samples}")
        
        """ if delta2_all:  # 防止空
            delta2_arr = np.array(delta2_all)
            self.delta2_history.append(delta2_arr)  # 保存历史
            msg = (f"[Delta2] mean={delta2_arr.mean():.4f}, std={delta2_arr.std():.4f}, "
           f"min={delta2_arr.min():.4f}, max={delta2_arr.max():.4f}\n")
            print(msg)
            if self.log_path is not None:
                with open(self.log_path, 'a') as log_file:
                    log_file.write(msg + "\n") """

        """ if loss_all:  # 防止空
            loss_arr = np.array(loss_all)
            msg = (f"[loss] mean={loss_arr.mean():.4f}, std={loss_arr.std():.4f}, "
           f"min={loss_arr.min():.4f}, max={loss_arr.max():.4f}\n")
            print(msg)
            if self.log_path is not None:
                with open(self.log_path, 'a') as log_file:
                    log_file.write(msg + "\n") """

        """ if self.shake_indices:
            delta2_arr = np.array(delta2_all)
            # 取 shake_indices 部分
            shake_mask = np.isin(np.arange(len(delta2_arr)), list(self.shake_indices))
            delta2_shake = delta2_arr[shake_mask]
            self.shake_delta2_history.append(delta2_shake)  # 保存历史
            # 打印 shake_indices 部分统计
            if len(delta2_shake) > 0:
                msg_shake = (f"[Delta2-SHAKE] mean={delta2_shake.mean():.4f}, std={delta2_shake.std():.4f}, "
                            f"min={delta2_shake.min():.4f}, max={delta2_shake.max():.4f}")
                print(msg_shake)
            else:
                msg_shake = "[Delta2-SHAKE] no indices found"

            # 写日志
            if self.log_path is not None:
                with open(self.log_path, 'a') as log_file:
                    log_file.write(msg_shake + "\n") """
        
        
        


    def step_v3(self):
        """
        在一个 epoch 或一个大 step 结束后调用，更新 removed 集合. 全新集合，不包含之前的
        """
        new_removed, new_restore = set(), set()
        new_shake, restore_shake = set(), set()
        delta2_all = [] 
        loss_all= []

        for idx in range(self.num_samples):
            if len(self.loss_history[idx]) < self.window:
                continue
            # 计算二阶差分
            c = self.loss_history[idx]
            delta2 = c[-1] - 2 * c[-2] + c[-3]
            delta2_all.append(abs(delta2))
            stable_label = len(set(self.label_history[idx])) == 1
            mean_loss = sum(c) / len(c)
            loss_all.append(mean_loss)
            high_stability = mean_loss < 0.2

            if abs(delta2) < self.delta_thresh and stable_label and high_stability:
                new_removed.add(idx)

            if abs(delta2) > self.shake_thresh:
                new_shake.add(idx)

        # 更新 removed 集合
        self.removed =new_removed
        self.shake = (self.shake | new_shake) - restore_shake

        # 记录恢复日志
        if len(new_restore) > 0:
            self.restore_log.append((len(self.restore_log), list(new_restore)))

        # 🔥 输出当前可参与反向传播的样本数
        num_active = self.num_samples - len(self.removed)
        print(f"Active samples for backprop: {num_active}/{self.num_samples} (removed {len(self.removed)})")
        print(f"samples shake: {len(self.shake)}/{self.num_samples}")

        
        if delta2_all:  # 防止空
            delta2_arr = np.array(delta2_all)
            self.delta2_history.append(delta2_arr)  # 保存历史
            msg = (f"[Delta2] mean={delta2_arr.mean():.4f}, std={delta2_arr.std():.4f}, "
           f"min={delta2_arr.min():.4f}, max={delta2_arr.max():.4f}\n")
            print(msg)
            if self.log_path is not None:
                with open(self.log_path, 'a') as log_file:
                    log_file.write(msg + "\n")

        if loss_all:  # 防止空
            loss_arr = np.array(loss_all)
            msg = (f"[loss] mean={loss_arr.mean():.4f}, std={loss_arr.std():.4f}, "
           f"min={loss_arr.min():.4f}, max={loss_arr.max():.4f}\n")
            print(msg)

    def filter_batch(self, batch_indices, mask):
        """
        过滤 batch 中的样本，跳过 removed 的
        Args:
            batch_indices (Tensor): 当前 batch 的样本全局索引
            mask (Tensor[bool]): 原始选择 mask (比如 selected_idx)
        Returns:
            mask (Tensor[bool]): 更新后的 mask
        """
        
        #cur_epoch = len(self.delta2_history)
                  
        device = mask.device
        keep_mask = torch.tensor(
            [idx.item() not in self.removed for idx in batch_indices],
            device=device, dtype=torch.bool
        )
        
        return mask & keep_mask
    
    def get_uncertainty_weights(self, batch_indices):
        """
        基于强弱一致性计算动态权重
        """
        weights = []
        for idx in batch_indices:
            idx = idx.item()
            if len(self.loss_history[idx]) < self.window:
                weights.append(1.0)
            else:
                c = self.loss_history[idx]
                delta2 = c[-1] - 2 * c[-2] + c[-3]
                #delta2 = delta2*len(delta2)/(delta2.sum()+1e-6)
                #u = self.loss_history[idx][-1]  # 最新不确定性
                #w = np.exp(-self.beta * (delta2 - self.mu)**2)
                if delta2 > 1:
                    w= delta2
                else:
                    w=1
                weights.append(w)

        return torch.tensor(weights, device='cuda', dtype=torch.float32)
    
    def plot_delta2_distributions(self, bins=50, interval=10):
        """
        每 interval 个 epoch 绘制一次 delta2 直方图，并保存到 cfg['cdc_dir'] 下
        """
        
        save_dir = self.cfg['cdc_dir']
        #os.makedirs(save_dir, exist_ok=True)

        for epoch, delta2_arr in enumerate(self.delta2_history, start=1):
            if epoch % interval == 0:  # 每 interval 个 epoch 画一次
                plt.figure()
                plt.hist(delta2_arr, bins=bins, alpha=0.7, color="blue")
                plt.title(f"Delta2 Distribution - Epoch {epoch}")
                plt.xlabel("Delta2")
                plt.ylabel("Count")

                save_path = os.path.join(save_dir, f"delta2_dist_epoch{epoch}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"✅ Saved delta2 distribution at {save_path}")
            
    def plot_delta2_trend(self):
        """绘制 delta2 均值/标准差 随 epoch 变化的曲线"""
        
        means = [arr.mean() for arr in self.delta2_history]
        stds  = [arr.std() for arr in self.delta2_history]
        
        save_dir = self.cfg['cdc_dir']

        plt.figure()
        plt.plot(means, label="mean Δ²", marker="o")
        plt.plot(stds, label="std Δ²", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title("Delta2 Trend over Epochs")
        plt.legend()
        save_path = os.path.join(save_dir, f"delta2_trend.png")
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved delta2 trend at {save_path}")
    


import wandb
import torch.nn.functional as F
from cdc.utils.torch_clustering import PyTorchKMeans
import time
def train_cali_sample(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, tracker:SampleMasterTracker, consisloss=False, stabilityloss=False, weight = False):
    
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    epoch_start = time.time()   # 开始计时
    for step, batch in enumerate(train_dataloader):     
        model.zero_grad()
        optimizer_all.zero_grad()
        #st = time.time()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        #gt = batch['target'].cuda(non_blocking=True)
        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]
            # 计算原始图像和增强图像的特征
            feature_weak = model(images, forward_pass='backbone')
            feature_augmented = model(images_augmented, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')
            
        # 计算特征稳定性指标
        feature_stability = F.cosine_similarity(feature_weak, feature_augmented, dim=1)
        stability_loss = 1 - feature_stability  # 转换为损失形式，越小表示越稳定
        feature_norm1 = F.normalize(feature_val, p=1, dim=1)

        #clu_softmax = F.softmax(output_clu, dim=1)
        cali_softmax = F.softmax(output_cali, dim=1)
        #clu_prob, clu_label = torch.max(clu_softmax, dim=1)
        cali_prob, cali_label = torch.max(cali_softmax, dim=1)

        #num_classes = output_cali.size(1)
        #cali_onehot = F.one_hot(cali_label, num_classes=num_classes).float()
        #cali_ce_loss = F.cross_entropy(output_cali, cali_onehot, reduction='none')
        #pdb.set_trace()

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

        #cluster_consistency_loss = (-super_target * F.log_softmax(output_cali)).sum(1)

        sub_steps = int(cfg['optimizer']['batch_size']/cfg['optimizer']['sub_batch_size'])
        sub_idxs = torch.range(0, sub_steps*cfg['optimizer']['sub_batch_size']-1).to(torch.int64).reshape(sub_steps,-1)
        for sub_step in range(sub_steps):
            sub_idx = sub_idxs[sub_step]
            output_aug = model(images_augmented[sub_idx])[0]
            sub_proto_pseudo, sub_selected_idx = proto_pseudo[sub_idx], selected_idx[sub_idx]

            # 过滤掉 mastered 样本
            mask = tracker.filter_batch(images_index[sub_idx], sub_selected_idx)
            if mask.sum() == 0:
                continue

            loss_ce = F.cross_entropy(output_aug[mask], sub_proto_pseudo[mask])
            loss = loss_ce
            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss.detach())

            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()

            output_cali = cali_mlp(feature_val[sub_idx], forward_pass='calibration')
            output_cali = output_cali[mask]
            cali_prob, _ = F.softmax(output_cali, dim=1).max(1)

            loss_cos = (-super_target[sub_idx][mask]*F.log_softmax(output_cali)).sum(1).mean()
            x_ = torch.mean(F.softmax(output_cali, dim=1), 0)
            loss_entropy = torch.sum(x_ * torch.log(x_))

            loss = loss_cos+cfg['method_kwargs']['w_en']*loss_entropy

            loss_cali.append(loss.detach())
            loss_coss.append(loss_cos.detach())
            loss_ens.append(loss_entropy.detach())

            optimizer_cali.zero_grad()
            loss.backward()
            optimizer_cali.step()

        # update tracker

        tracker.update(
                indices=images_index.tolist(),
                confidences=cali_prob.tolist(),
                labels=cali_label.tolist(),
                losses=stability_loss.tolist()
            )
        """ if consisloss:
            #pdb.set_trace()
            tracker.update(
                indices=images_index.tolist(),
                confidences=cali_prob.tolist(),
                labels=cali_label.tolist(),
                losses=cluster_consistency_loss.tolist()
            )

        elif stabilityloss:
            tracker.update(
                indices=images_index.tolist(),
                confidences=cali_prob.tolist(),
                labels=cali_label.tolist(),
                losses=stability_loss.tolist()
            )

        else:
            tracker.update(
                indices=images_index.tolist(),
                confidences=cali_prob.tolist(),
                labels=cali_label.tolist(),
                losses=cali_ce_loss.tolist()
            ) """


    epoch_time = time.time() - epoch_start
    tracker.step()
    #print(f"Removed={len(tracker.removed)}, restored={len(tracker.restore_log)}")
    

    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })

    return epoch_time

import time


def train_cali_sample_time(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, tracker:SampleMasterTracker, consisloss=False, stabilityloss=False, weight = False):
    
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    epoch_start = time.time()   # 开始计时

    # 记录时间
    time_dataloader, time_forward, time_loss, time_backward, time_step = 0,0,0,0,0

    for step, batch in enumerate(train_dataloader):     
        batch_start = time.perf_counter()

        # ---- DataLoader 部分 ----
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        time_dataloader += time.perf_counter() - batch_start

        # ---- Forward ----
        torch.cuda.synchronize()
        start_forward = time.perf_counter()

        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]

            feature_weak = model(images, forward_pass='backbone')
            feature_augmented = model(images_augmented, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')

        torch.cuda.synchronize()
        time_forward += time.perf_counter() - start_forward

        # ---- Loss 准备部分 ----
        torch.cuda.synchronize()
        start_loss = time.perf_counter()

        feature_stability = F.cosine_similarity(feature_weak, feature_augmented, dim=1)
        stability_loss = 1 - feature_stability  
        feature_norm1 = F.normalize(feature_val, p=1, dim=1)

        cali_softmax = F.softmax(output_cali, dim=1)
        cali_prob, cali_label = torch.max(cali_softmax, dim=1)

        proto_pseudo = cali_label
        selected_num = cfg['method_kwargs']['per_class_selected_num']
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

        # ---- Sub-batch 训练循环 ----
        sub_steps = int(cfg['optimizer']['batch_size']/cfg['optimizer']['sub_batch_size'])
        sub_idxs = torch.arange(0, sub_steps*cfg['optimizer']['sub_batch_size']).to(torch.int64).reshape(sub_steps,-1)

        for sub_step in range(sub_steps):
            sub_idx = sub_idxs[sub_step]

            torch.cuda.synchronize()
            start_forward2 = time.perf_counter()
            output_aug = model(images_augmented[sub_idx])[0]
            torch.cuda.synchronize()
            time_forward += time.perf_counter() - start_forward2

            sub_proto_pseudo, sub_selected_idx = proto_pseudo[sub_idx], selected_idx[sub_idx]

            mask = tracker.filter_batch(images_index[sub_idx], sub_selected_idx)
            if mask.sum() == 0:
                continue

            torch.cuda.synchronize()
            start_loss2 = time.perf_counter()
            loss_ce = F.cross_entropy(output_aug[mask], sub_proto_pseudo[mask])
            loss = loss_ce
            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss.detach())
            torch.cuda.synchronize()
            time_loss += time.perf_counter() - start_loss2

            # ---- backward + step (optimizer_all) ----
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

            # ---- calibration 分支 ----
            torch.cuda.synchronize()
            start_forward3 = time.perf_counter()
            output_cali = cali_mlp(feature_val[sub_idx], forward_pass='calibration')
            output_cali = output_cali[mask]
            cali_prob, _ = F.softmax(output_cali, dim=1).max(1)
            torch.cuda.synchronize()
            time_forward += time.perf_counter() - start_forward3

            torch.cuda.synchronize()
            start_loss3 = time.perf_counter()
            loss_cos = (-super_target[sub_idx][mask]*F.log_softmax(output_cali)).sum(1).mean()
            x_ = torch.mean(F.softmax(output_cali, dim=1), 0)
            loss_entropy = torch.sum(x_ * torch.log(x_))
            loss = loss_cos+cfg['method_kwargs']['w_en']*loss_entropy

            loss_cali.append(loss.detach())
            loss_coss.append(loss_cos.detach())
            loss_ens.append(loss_entropy.detach())
            torch.cuda.synchronize()
            time_loss += time.perf_counter() - start_loss3

            # ---- backward + step (optimizer_cali) ----
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

        # ---- 更新 tracker ----
        tracker.update(
                indices=images_index.tolist(),
                confidences=cali_prob.tolist(),
                labels=cali_label.tolist(),
                losses=stability_loss.tolist()
            )

    epoch_time = time.time() - epoch_start
    tracker.step()

    # 打印耗时结果
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

    return epoch_time

def train_cali_sample_speed(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, tracker:SampleMasterTracker, consisloss=False, stabilityloss=False, weight=False):

    loss_clu, loss_cali = [], []
    loss_ces, loss_ens, loss_coss = [], [], []
    epoch_start = time.time()   # 开始计时

    add_time = time.time()
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        feature_weak = model(images, forward_pass='backbone')
        feature_augmented = model(images_augmented, forward_pass='backbone')

    feature_stability = F.cosine_similarity(feature_weak, feature_augmented, dim=1)
    stability_loss = 1 - feature_stability

    tracker.update(
            indices=images_index.tolist(),
            confidences=cali_prob.tolist(),
            labels=cali_label.tolist(),
            losses=stability_loss.tolist()
        )
    add = time.time() - add_time
    print("add: ", add)
    

    # 记录各环节时间
    time_dataloader, time_forward, time_loss, time_backward, time_step = 0, 0, 0, 0, 0

    for step, batch in enumerate(train_dataloader):
        batch_start = time.perf_counter()

        # ------------------- DataLoader -------------------
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        time_dataloader += time.perf_counter() - batch_start

        # ------------------- Forward (full batch) -------------------
        torch.cuda.synchronize()
        start_forward = time.perf_counter()

        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]

            feature_weak = model(images, forward_pass='backbone')
            feature_augmented = model(images_augmented, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')
        
        

        torch.cuda.synchronize()
        time_forward += time.perf_counter() - start_forward

        # ------------------- Loss 准备 -------------------
        torch.cuda.synchronize()
        start_loss = time.perf_counter()

        feature_stability = F.cosine_similarity(feature_weak, feature_augmented, dim=1)
        stability_loss = 1 - feature_stability

        

        feature_norm1 = F.normalize(feature_val, p=1, dim=1)

        cali_softmax = F.softmax(output_cali, dim=1)
        cali_prob, cali_label = torch.max(cali_softmax, dim=1)

        proto_pseudo = cali_label
        selected_num = cfg['method_kwargs']['per_class_selected_num']
        selected_idx = torch.zeros(len(cali_softmax)).cuda()
        for label_idx in range(output_clu.shape[1]):
            per_label_mask = cali_softmax[:, label_idx].sort(descending=True)[1][:selected_num]
            sel = int(cali_prob[per_label_mask].mean() * selected_num)
            selected_idx[per_label_mask[:sel]] = 1
        selected_idx = selected_idx == 1

        cluster_num = cfg['method_kwargs']['super_cluster_num']
        KMeans_all = PyTorchKMeans(init='k-means++', n_clusters=cluster_num, verbose=False)
        split_all = KMeans_all.fit_predict(feature_norm1)
        target_dict = torch.stack([F.softmax(output_clu_val, dim=1)[split_all == i].mean(0) for i in range(cluster_num)])
        super_target = target_dict[split_all]

        torch.cuda.synchronize()
        time_loss += time.perf_counter() - start_loss

        # ------------------- Sub-batch 训练 -------------------
        sub_steps = int(cfg['optimizer']['batch_size'] / cfg['optimizer']['sub_batch_size'])
        sub_idxs = torch.arange(0, sub_steps * cfg['optimizer']['sub_batch_size']).to(torch.int64).reshape(sub_steps, -1)

        for sub_step in range(sub_steps):
            sub_idx = sub_idxs[sub_step]

            sub_images = images_augmented[sub_idx]
            sub_proto_pseudo = proto_pseudo[sub_idx]
            sub_selected_idx = selected_idx[sub_idx]
            sub_index = images_index[sub_idx]

            # ---- 提前 mask ----
            mask = tracker.filter_batch(sub_index, sub_selected_idx)
            if mask.sum() == 0:
                continue
            sub_images = sub_images[mask]
            sub_proto_pseudo = sub_proto_pseudo[mask]
            sub_index = sub_index[mask]

            # ------------------- Forward -------------------
            torch.cuda.synchronize()
            start_forward2 = time.perf_counter()
            output_aug = model(sub_images)[0]
            torch.cuda.synchronize()
            time_forward += time.perf_counter() - start_forward2

            # ------------------- Loss (clu) -------------------
            torch.cuda.synchronize()
            start_loss2 = time.perf_counter()
            loss_ce = F.cross_entropy(output_aug, sub_proto_pseudo)
            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss_ce.detach())
            torch.cuda.synchronize()
            time_loss += time.perf_counter() - start_loss2

            # ------------------- Backward + Step (optimizer_all) -------------------
            optimizer_all.zero_grad()
            torch.cuda.synchronize()
            start_backward = time.perf_counter()
            loss_ce.backward()
            torch.cuda.synchronize()
            time_backward += time.perf_counter() - start_backward

            torch.cuda.synchronize()
            start_step = time.perf_counter()
            optimizer_all.step()
            torch.cuda.synchronize()
            time_step += time.perf_counter() - start_step

            # ------------------- Calibration -------------------
            feature_val_sub = feature_val[sub_idx][mask]

            torch.cuda.synchronize()
            start_forward3 = time.perf_counter()
            output_cali = cali_mlp(feature_val_sub, forward_pass='calibration')
            cali_prob, _ = F.softmax(output_cali, dim=1).max(1)
            torch.cuda.synchronize()
            time_forward += time.perf_counter() - start_forward3

            torch.cuda.synchronize()
            start_loss3 = time.perf_counter()
            loss_cos = (-super_target[sub_idx][mask] * F.log_softmax(output_cali)).sum(1).mean()
            x_ = torch.mean(F.softmax(output_cali, dim=1), 0)
            loss_entropy = torch.sum(x_ * torch.log(x_))
            loss = loss_cos + cfg['method_kwargs']['w_en'] * loss_entropy

            loss_cali.append(loss.detach())
            loss_coss.append(loss_cos.detach())
            loss_ens.append(loss_entropy.detach())
            torch.cuda.synchronize()
            time_loss += time.perf_counter() - start_loss3

            # ------------------- Backward + Step (optimizer_cali) -------------------
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

        # ------------------- 更新 tracker -------------------
        tracker.update(
            indices=images_index.tolist(),
            confidences=cali_prob.tolist(),
            labels=cali_label.tolist(),
            losses=stability_loss.tolist()
        )

    epoch_time = time.time() - epoch_start
    tracker.step()

    # 打印耗时
    print(f"[Time] DataLoader: {time_dataloader:.3f}s, "
          f"Forward: {time_forward:.3f}s, "
          f"Loss: {time_loss:.3f}s, "
          f"Backward: {time_backward:.3f}s, "
          f"Step: {time_step:.3f}s, "
          f"Total epoch: {epoch_time:.3f}s")

    wandb.log({
        "loss_clu": torch.stack(loss_clu).mean(),
        "loss_cali": torch.stack(loss_cali).mean(),
        "loss_ces": torch.stack(loss_ces).mean(),
        "loss_cos": torch.stack(loss_coss).mean(),
        "loss_ens": torch.stack(loss_ens).mean(),
    })

    return epoch_time

def train_cali_sample_amp(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, tracker, 
                          consisloss=False, stabilityloss=False, weight=False, log_interval=50):
    scaler = torch.cuda.amp.GradScaler()

    loss_clu, loss_cali = [], []
    loss_ces, loss_ens, loss_coss = [], [], []

    epoch_start = time.time()

    for step, batch in enumerate(train_dataloader):
        torch.cuda.synchronize()
        t0 = time.time()

        # ------------------- Data Loading -------------------
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)

        torch.cuda.synchronize()
        t_data = time.time()

        model.train()
        cali_mlp.train()

        with torch.no_grad(), torch.cuda.amp.autocast():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]
            feature_weak = model(images, forward_pass='backbone')
            feature_augmented = model(images_augmented, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')

        torch.cuda.synchronize()
        t_forward = time.time()

        # ------------------- Loss Prep -------------------
        with torch.cuda.amp.autocast():
            feature_stability = F.cosine_similarity(feature_weak, feature_augmented, dim=1)
            stability_loss = 1 - feature_stability
            feature_norm1 = F.normalize(feature_val, p=1, dim=1)

            cali_softmax = F.softmax(output_cali, dim=1)
            cali_prob, cali_label = torch.max(cali_softmax, dim=1)

        # ✅ 正确构造布尔型 selected_idx
        selected_num = cfg['method_kwargs']['per_class_selected_num']
        selected_idx = torch.zeros(len(cali_softmax), dtype=torch.bool, device=cali_softmax.device)

        for label_idx in range(output_clu.shape[1]):
            per_label_mask = cali_softmax[:, label_idx].sort(descending=True)[1][:selected_num]
            sel = int(cali_prob[per_label_mask].mean() * selected_num)
            selected_idx[per_label_mask[:sel]] = True  # ✅ 布尔赋值

        torch.cuda.synchronize()
        t_loss_prep = time.time()

        # ------------------- Training Sub-Steps -------------------
        sub_steps = int(cfg['optimizer']['batch_size']/cfg['optimizer']['sub_batch_size'])
        sub_idxs = torch.arange(0, sub_steps*cfg['optimizer']['sub_batch_size']).to(torch.int64).reshape(sub_steps,-1)

        for sub_step in range(sub_steps):
            sub_idx = sub_idxs[sub_step]

            with torch.cuda.amp.autocast():
                output_aug = model(images_augmented[sub_idx])[0]
                sub_proto_pseudo = cali_label[sub_idx]
                sub_selected_idx = selected_idx[sub_idx]  # ✅ bool 类型

                mask = tracker.filter_batch(images_index[sub_idx], sub_selected_idx)
                if mask.sum() == 0:
                    continue

                loss_ce = F.cross_entropy(output_aug[mask], sub_proto_pseudo[mask])
                loss = loss_ce

            optimizer_all.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer_all)
            scaler.update()

            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss.detach())

        torch.cuda.synchronize()
        t_backward = time.time()

        # ------------------- Tracker 更新 -------------------
        tracker.update(
            indices=images_index.tolist(),
            confidences=cali_prob.tolist(),
            labels=cali_label.tolist(),
            losses=stability_loss.tolist()
        )

        torch.cuda.synchronize()
        t_step = time.time()

        if (step+1) % log_interval == 0:
            print(
                f"[Step {step+1}] "
                f"Data: {t_data-t0:.4f}s | Forward: {t_forward-t_data:.4f}s | "
                f"LossPrep: {t_loss_prep-t_forward:.4f}s | Backward: {t_backward-t_loss_prep:.4f}s | "
                f"Step/Update: {t_step-t_backward:.4f}s"
            )

    epoch_time = time.time() - epoch_start
    tracker.step()

    wandb.log({
        "loss_clu": torch.stack(loss_clu).mean().item(),
        "loss_cali": torch.stack(loss_cali).mean().item() if loss_cali else 0,
        "loss_ces": torch.stack(loss_ces).mean().item() if loss_ces else 0,
        "loss_cos": torch.stack(loss_coss).mean().item() if loss_coss else 0,
        "loss_ens": torch.stack(loss_ens).mean().item() if loss_ens else 0,
    })

    print(f"[Epoch Done] Total time: {epoch_time:.2f}s")

    return epoch_time

def train_cali_sample_v2(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, tracker:SampleMasterTracker, consisloss=False, stabilityloss=False, weight = False):
    
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    print(f"[DataLoader] Active samples this epoch: {len(train_dataloader.sampler)}")

    epoch_start = time.time()   # 开始计时

    for step, batch in enumerate(train_dataloader):     
        model.zero_grad()
        optimizer_all.zero_grad()
        #st = time.time()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        #gt = batch['target'].cuda(non_blocking=True)
        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]
            # 计算原始图像和增强图像的特征
            feature_weak = model(images, forward_pass='backbone')
            feature_augmented = model(images_augmented, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')
            
        # 计算特征稳定性指标
        feature_stability = F.cosine_similarity(feature_weak, feature_augmented, dim=1)
        stability_loss = 1 - feature_stability  # 转换为损失形式，越小表示越稳定
        feature_norm1 = F.normalize(feature_val, p=1, dim=1)

        cali_softmax = F.softmax(output_cali, dim=1)
        cali_prob, cali_label = torch.max(cali_softmax, dim=1)

        proto_pseudo = cali_label
        selected_num = cfg['method_kwargs']['per_class_selected_num']
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

            if sub_selected_idx.sum() == 0:
                continue

            loss_ce = F.cross_entropy(output_aug[sub_selected_idx], sub_proto_pseudo[sub_selected_idx])
            loss = loss_ce
            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss.detach())

            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()

            output_cali = cali_mlp(feature_val[sub_idx], forward_pass='calibration')
            output_cali = output_cali
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

        # update tracker
        tracker.update(
                indices=images_index.tolist(),
                confidences=cali_prob.tolist(),
                labels=cali_label.tolist(),
                losses=stability_loss.tolist()
            )

    epoch_time = time.time() - epoch_start
    
    tracker.step()


    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })

    return epoch_time
    

def train_cali_sample_v3(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, tracker:SampleMasterTracker, consisloss=False, stabilityloss=False, weight = False):
    
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    epoch_start = time.time()   # 开始计时
    for step, batch in enumerate(train_dataloader):     
        model.zero_grad()
        optimizer_all.zero_grad()
        #st = time.time()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        #gt = batch['target'].cuda(non_blocking=True)
        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]
            # 计算原始图像和增强图像的特征
            feature_weak = model(images, forward_pass='backbone')
            feature_augmented = model(images_augmented, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')
            
        # 计算特征稳定性指标
        feature_stability = F.cosine_similarity(feature_weak, feature_augmented, dim=1)
        stability_loss = 1 - feature_stability  # 转换为损失形式，越小表示越稳定
        feature_norm1 = F.normalize(feature_val, p=1, dim=1)

        cali_softmax = F.softmax(output_cali, dim=1)
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

        #cluster_consistency_loss = (-super_target * F.log_softmax(output_cali)).sum(1)

        sub_steps = int(cfg['optimizer']['batch_size']/cfg['optimizer']['sub_batch_size'])
        sub_idxs = torch.range(0, sub_steps*cfg['optimizer']['sub_batch_size']-1).to(torch.int64).reshape(sub_steps,-1)
        for sub_step in range(sub_steps):
            sub_idx = sub_idxs[sub_step]
            output_aug = model(images_augmented[sub_idx])[0]
            sub_proto_pseudo, sub_selected_idx = proto_pseudo[sub_idx], selected_idx[sub_idx]

            loss_ce = F.cross_entropy(output_aug, sub_proto_pseudo)
            loss = loss_ce
            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss.detach())

            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()

            output_cali = cali_mlp(feature_val[sub_idx], forward_pass='calibration')
            output_cali = output_cali
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

        # update tracker

        tracker.update(
                indices=images_index.tolist(),
                confidences=cali_prob.tolist(),
                labels=cali_label.tolist(),
                losses=stability_loss.tolist()
            )


    epoch_time = time.time() - epoch_start
    tracker.step_v3()
    
    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })

    return epoch_time



def train_cali_sample_su(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all):
    
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    epoch_start = time.time()   # 开始计时
    for step, batch in enumerate(train_dataloader):     
        model.zero_grad()
        optimizer_all.zero_grad()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]
            # 计算原始图像和增强图像的特征
            feature_weak = model(images, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')
            

        feature_norm1 = F.normalize(feature_val, p=1, dim=1)

        cali_softmax = F.softmax(output_cali, dim=1)
        cali_prob, cali_label = torch.max(cali_softmax, dim=1)

        proto_pseudo = cali_label
        selected_num = cfg['method_kwargs']['per_class_selected_num']

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

            loss_ce = F.cross_entropy(output_aug, sub_proto_pseudo)
            loss = loss_ce
            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss.detach())

            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()

            output_cali = cali_mlp(feature_val[sub_idx], forward_pass='calibration')
            output_cali = output_cali
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

    epoch_time = time.time() - epoch_start
    
    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })

    return epoch_time



def train_cali(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, epoch, start_epoch):
    loss_clu, loss_cali = [],[]
    loss_ces, loss_ens, loss_coss = [],[],[]
    epoch_start = time.time()   # 开始计时
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        optimizer_all.zero_grad()
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

    epoch_time = time.time() - epoch_start
    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })
    return epoch_time