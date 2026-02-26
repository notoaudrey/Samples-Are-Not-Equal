import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

class DynamicUncertaintySelector:
    def __init__(self, num_samples, window_size=5, start_epoch=10, keep_ratio=0.6, update=1):
        self.num_samples = num_samples
        self.window_size = window_size  # k
        self.start_epoch = start_epoch  # J
        self.keep_ratio = keep_ratio
        self.updatef = update

        # 初始化滑动窗口 [N, k]
        self.pred_history = torch.zeros(num_samples, window_size).cuda()
        self.ptr = torch.zeros(num_samples, dtype=torch.long).cuda()  # 每个样本的写入位置
        self.epoch = 0

    def update(self, sample_indices, logits):
        """
        更新当前 epoch 的预测概率到滑动窗口中。
        Args:
            sample_indices: Tensor [B]，表示当前 batch 中样本在全集中的 index
            pred_probs: Tensor [B]，当前 batch 中每个样本对其真实标签的预测概率
        """
        
        probs = torch.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1).values  # [B]
        
        for i, idx in enumerate(sample_indices):
            pos = self.ptr[idx].item()
            self.pred_history[idx, pos] = max_probs[i]
            self.ptr[idx] = (pos + 1) % self.window_size

    def select(self):
        """
        计算标准差作为伪标签置信度的波动 → 用作样本不确定性
        Returns:
            keep_indices: Tensor [K]，保留样本索引（Top-K 不确定性）
        """
        self.epoch += 1
        if self.epoch < self.start_epoch:
            return torch.arange(self.num_samples).cuda()  # 返回所有样本索引

        with torch.no_grad():
            uncertainties = torch.std(self.pred_history, dim=1)  # shape: [N]
            num_keep = int(self.keep_ratio * self.num_samples)
            topk = torch.topk(uncertainties, num_keep)
            keep_indices = topk.indices
            return keep_indices
        


class DynamicEntropySelector:
    def __init__(self, num_samples, window_size=5, start_epoch=10, keep_ratio=0.6, update=1):
        self.num_samples = num_samples
        self.window_size = window_size
        self.start_epoch = start_epoch
        self.keep_ratio = keep_ratio
        self.updatef = update

        self.entropy_history = torch.zeros(num_samples, window_size).cuda()
        self.pseudo_history = torch.zeros(num_samples, window_size).cuda()
        self.conf_history = torch.zeros(num_samples, window_size).cuda()
        self.ptr = torch.zeros(num_samples, dtype=torch.long).cuda()
        self.epoch = 0

    def update(self, sample_indices, logits):
        """
        使用每个样本的 softmax 熵作为不确定性，加入滑动窗口。
        Args:
            sample_indices: Tensor [B]
            logits: Tensor [B, C]
        """
        probs = torch.softmax(logits, dim=1) + 1e-8  # 避免 log(0)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)  # shape: [B]

        for i, idx in enumerate(sample_indices):
            pos = self.ptr[idx].item()
            self.entropy_history[idx, pos] = entropy[i]
            self.pseudo_history[idx, pos] = probs[i].argmax().item()  # 记录伪标签
            self.conf_history[idx, pos] = probs[i].max().item()
            self.ptr[idx] = (pos + 1) % self.window_size

        

    def select(self):
        """
        返回动态熵不确定性（std over entropy）最高的 top-k 样本索引
        """
        self.epoch += 1
        
        if self.epoch < self.start_epoch:
            return torch.arange(self.num_samples).cuda()

        with torch.no_grad():
            uncertainties = torch.std(self.entropy_history, dim=1)  # shape: [N]
            num_keep = int(self.keep_ratio * self.num_samples)
            topk = torch.topk(uncertainties, num_keep)
            return topk.indices
        
        
class DynamicPseudoSelector:
    def __init__(self, num_samples, window_size=5, start_epoch=10, keep_ratio=0.6, update=1, mean=10):
        self.num_samples = num_samples
        self.window_size = window_size
        self.start_epoch = start_epoch
        self.keep_ratio = keep_ratio
        self.updatef = update
        self.mean = mean

        self.entropy_history = torch.zeros(num_samples, window_size).cuda()
        self.pseudo_history = torch.zeros(num_samples, window_size).cuda()
        self.conf_history = torch.zeros(num_samples, window_size).cuda()
        self.ptr = torch.zeros(num_samples, dtype=torch.long).cuda()
        self.epoch = 0

    def update(self, sample_indices, logits):
        """
        使用每个样本的 softmax 熵作为不确定性，加入滑动窗口。
        Args:
            sample_indices: Tensor [B]
            logits: Tensor [B, C]
        """
        probs = torch.softmax(logits, dim=1) + 1e-8  # 避免 log(0)
        entropy = -torch.sum(probs * torch.log(probs), dim=1)  # shape: [B]

        for i, idx in enumerate(sample_indices):
            pos = self.ptr[idx].item()
            self.entropy_history[idx, pos] = entropy[i]
            self.pseudo_history[idx, pos] = probs[i].argmax().item()  # 记录伪标签
            self.conf_history[idx, pos] = probs[i].max().item()
            self.ptr[idx] = (pos + 1) % self.window_size

        

    def select(self):
        """
        返回动态熵不确定性（std over entropy）最高的 top-k 样本索引
        """
        self.epoch += 1
        
        if self.epoch < self.start_epoch:
            return torch.arange(self.num_samples).cuda()

        with torch.no_grad():
            uncertainties = torch.std(self.entropy_history, dim=1)  # shape: [N]
            
            label_changes = torch.zeros(self.num_samples).cuda()
            for i in range(self.window_size - 1):
                changed = self.pseudo_history[:, i] != self.pseudo_history[:, i + 1]
                valid = (self.pseudo_history[:, i] >= 0) & (self.pseudo_history[:, i + 1] >= 0)
                label_changes += (changed & valid).float()
            num_keep = int(self.keep_ratio * self.num_samples)
            selected = torch.topk(-label_changes, num_keep).indices  # 越小越好，取负号后用 topk
            return selected
        
        
    def select_consis(self, conf_threshold=0.99):
        """
        返回：伪标签不变且置信度高的样本索引
        Args:
            conf_threshold: float, 置信度阈值
        """
        self.epoch += 1

        if self.epoch < self.start_epoch:
            return torch.arange(self.num_samples).cuda()

        with torch.no_grad():
            # --------- 条件 1: 伪标签不变 ----------
            # 每个样本在窗口中的伪标签
            pseudo_seq = self.pseudo_history  # [N, W]
            # 检查每行的伪标签是否全相同
            same_label = (pseudo_seq == pseudo_seq[:, 0:1]).all(dim=1)  

            # --------- 条件 2: 高置信度 ----------
            # 这里用平均置信度，也可以换成最后一次
            if self.mean>0:
                mean_conf = self.conf_history[:, -self.mean:].mean(dim=1)
                cur_conf = mean_conf
            else:
                cur_conf = self.conf_history[:, -1]  # 最后一次的置信度
                  
            high_conf = cur_conf >= conf_threshold  

            # --------- 综合条件 ----------
            mask = same_label & high_conf
            selected = torch.nonzero(mask, as_tuple=False).squeeze(1)  
            
            sorted_idx = torch.argsort(selected, descending=True)  
             # --------- 截断保留 ----------
            max_keep = int((1-self.keep_ratio) * self.num_samples)
            keep_num = min(len(selected), max_keep)
            selected = selected[sorted_idx[:keep_num]]

            #import pdb; pdb.set_trace()
            
            return selected

def compute_accuracy_after_changes(selector, true_labels, target_epoch=5, change_count=1,throld=0.95, highcount=0):
    """
    计算在 target_epoch 时，伪标签变动次数=change_count 的样本准确率
    (使用匈牙利匹配对齐聚类标签和真标签)
    """
    N = selector.num_samples
    pseudo_hist = selector.pseudo_history[:N, :].clone().cpu()  # [N, window_size]

    # 转成 tensor
    if not isinstance(true_labels, torch.Tensor):
        true_labels = torch.tensor(true_labels)

    # 统计伪标签变动次数
    label_changes = torch.zeros(N)
    for i in range(selector.window_size - 1):
        changed = pseudo_hist[:, i] != pseudo_hist[:, i + 1]
        valid = (pseudo_hist[:, i] >= 0) & (pseudo_hist[:, i + 1] >= 0)
        label_changes += (changed & valid).float()

    # 选出伪标签变动次数==change_count 的样本
    selected_mask = (label_changes == change_count)
    selected_indices = torch.where(selected_mask)[0]

    if len(selected_indices) == 0:
        return 0.0, selected_indices

    # 取 target_epoch 的伪标签
    pseudo_labels = pseudo_hist[selected_indices, (target_epoch - 1) % selector.window_size].numpy()
    gt_labels = true_labels[selected_indices].cpu().numpy()

    # ---- 匈牙利匹配 ----
    num_clusters = max(pseudo_labels.max(), gt_labels.max()) + 1
    conf_mat = confusion_matrix(gt_labels, pseudo_labels, labels=range(int(num_clusters.item())))
    row_ind, col_ind = linear_sum_assignment(-conf_mat)  # 最大化匹配
    mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # 把聚类标签映射到真标签空间
    mapped_preds = [mapping[p] if p in mapping else p for p in pseudo_labels]
    #cifar10seed5{8: 0, 6: 1, 9: 2, 5: 3, 1: 4, 0: 5, 2: 6, 3: 7, 4: 8, 7: 9}
    #import pdb; pdb.set_trace()
    # ---- 计算准确率 ----
    acc = (torch.tensor(mapped_preds) == torch.tensor(gt_labels)).float().mean().item()
    
    
    pseudo_labels_all = pseudo_hist[:, (target_epoch - 1) % selector.window_size].numpy()
    mapped_preds_all = [mapping[p] if p in mapping else p for p in pseudo_labels_all]
    
    selected_mask0 = (label_changes == highcount)
    selected_indices0 = torch.where(selected_mask0)[0]
    last_conf_all = selector.conf_history[:, -1]
    last_conf_all0 = last_conf_all[selected_indices0]
    selected_indices0_high = selected_indices0[last_conf_all0.cpu()>throld]
    gt_labels0_high = true_labels[selected_indices0_high].cpu().numpy()
    pseudo_labels0_high = pseudo_hist[selected_indices0_high, (target_epoch - 1) % selector.window_size].numpy()
    mapped_preds0_high = [mapping[p] if p in mapping else p for p in pseudo_labels0_high]
    acc0_high = (torch.tensor(mapped_preds0_high) == torch.tensor(gt_labels0_high)).float().mean().item()
    print("highconf_consis_acc:", acc0_high, "high_num:", len(selected_indices0_high))
    
    import pdb; pdb.set_trace()
    
    return acc, selected_indices