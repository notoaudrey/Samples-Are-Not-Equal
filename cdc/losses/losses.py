import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import torch.distributed as dist
import math
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def gather(z):
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    #gather_z = diffdist.functional.all_gather(gather_z, z)
    gather_z = torch.cat(gather_z)

    return gather_z


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc


def mean_cumulative_gain(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    mcg = (topk == labels).float().mean(1)
    return mcg


def mean_average_precision(logits, labels, k):
    # TODO: not the fastest solution but looks fine
    argsort = torch.argsort(logits, dim=1, descending=True)
    labels_to_sorted_idx = (
        torch.sort(torch.gather(torch.argsort(argsort, dim=1), 1, labels), dim=1)[0] + 1
    )
    precision = (1 + torch.arange(k).float().cuda()) / labels_to_sorted_idx
    return precision.sum(1) / k

# CC
class CCInstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(CCInstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss
#for CC
class CCClusterLoss(nn.Module):
    def __init__(self, class_num, temperature):
        super(CCClusterLoss, self).__init__()
        self.class_num = class_num
        self.temperature = temperature

        self.mask = self.mask_correlated_clusters(class_num)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        p_i = torch.mean(c_i, dim=0)
        ne_i = np.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = torch.mean(c_j, dim=0)
        ne_j = np.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = F.cosine_similarity(c.unsqueeze(1), c.unsqueeze(0), dim=2) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = F.cross_entropy(logits, labels)

        return loss, ne_loss


#tcl loss
class InstanceLoss(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, tau=0.5, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed

    def forward(self, z, get_map=False):
        n = z.shape[0]
        assert n % self.multiplier == 0

        z = z / np.sqrt(self.tau)

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        
        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)

        return loss

class ClusterLoss(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, tau=1.0, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed

    def forward(self, c, get_map=False):
        n = c.shape[0]
        assert n % self.multiplier == 0

        # c = c / np.sqrt(self.tau)

        if self.distributed:
            c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            c_list = diffdist.functional.all_gather(c_list, c)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            c_list = [chunk for x in c_list for chunk in x.chunk(self.multiplier)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            c_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    c_sorted.append(c_list[i * self.multiplier + m])
            c_aug0 = torch.cat(
                c_sorted[: int(self.multiplier * dist.get_world_size() / 2)], dim=0
            )
            c_aug1 = torch.cat(
                c_sorted[int(self.multiplier * dist.get_world_size() / 2) :], dim=0
            )

            p_i = c_aug0.sum(0).view(-1)
            p_i /= p_i.sum()
            en_i = np.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
            p_j = c_aug1.sum(0).view(-1)
            p_j /= p_j.sum()
            en_j = np.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
            en_loss = en_i + en_j

            c = torch.cat((c_aug0.t(), c_aug1.t()), dim=0)
            n = c.shape[0]


        else:
            # If not using distributed training, simply work with the tensor `c` locally
            c_list = c.chunk(self.multiplier)  # Split c into `self.multiplier` chunks

            # Sort the chunks in the order we want
            c_sorted = []
            for m in range(self.multiplier):
                c_sorted.append(c_list[m])  # Collect the chunks in order

            # Now concatenate the chunks as in the original logic
            c_aug0 = torch.cat(c_sorted[: int(self.multiplier / 2)], dim=0)
            c_aug1 = torch.cat(c_sorted[int(self.multiplier / 2):], dim=0)

            # Compute probability distributions and their entropy
            p_i = c_aug0.sum(0).view(-1)
            p_i /= p_i.sum()  # Normalize to get a valid probability distribution
            en_i = np.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()  # Entropy calculation

            p_j = c_aug1.sum(0).view(-1)
            p_j /= p_j.sum()  # Normalize to get a valid probability distribution
            en_j = np.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()  # Entropy calculation

            # Total entropy loss
            en_loss = en_i + en_j
            
            # Concatenate the augmented labels (if necessary)
            c = torch.cat((c_aug0.t(), c_aug1.t()), dim=0)
            n = c.shape[0]  # Size of the concatenated tensor

        c = F.normalize(c, p=2, dim=1) / np.sqrt(self.tau)

        logits = c @ c.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n // m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        loss = -logprob[np.repeat(np.arange(n), m - 1), labels].sum() / n / (m - 1)
        
        return loss + en_loss

class InstanceLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(
        self,
        tau=0.5,
        multiplier=2,
        distributed=False,
        alpha=0.99,
        gamma=0.5,
        cluster_num=10,
    ):
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.alpha = alpha
        self.gamma = gamma
        self.cluster_num = cluster_num

    @torch.no_grad()
    def generate_pseudo_labels(self, c, pseudo_label_cur, index):
        if self.distributed:
            c_list = [torch.zeros_like(c) for _ in range(dist.get_world_size())]
            pseudo_label_cur_list = [torch.zeros_like(pseudo_label_cur) for _ in range(dist.get_world_size())]
            index_list = [torch.zeros_like(index) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            c_list = diffdist.functional.all_gather(c_list, c)
            pseudo_label_cur_list = diffdist.functional.all_gather(pseudo_label_cur_list, pseudo_label_cur)
            index_list = diffdist.functional.all_gather(index_list, index)
            c = torch.cat(c_list, dim=0,)
            pseudo_label_cur = torch.cat(pseudo_label_cur_list, dim=0,)
            index = torch.cat(index_list, dim=0,)
            
        else:
            c = torch.cat([c], dim=0)
            pseudo_label_cur = torch.cat([pseudo_label_cur], dim=0)
            index = torch.cat([index], dim=0)

        batch_size = c.shape[0]

        pseudo_label_nxt = -torch.ones(batch_size, dtype=torch.long).cuda()
        tmp = torch.arange(0, batch_size).cuda()
        
        prediction = c.argmax(dim=1)
        confidence = c.max(dim=1).values

        unconfident_pred_index = confidence < self.alpha
        pseudo_per_class = np.ceil(batch_size / self.cluster_num * self.gamma).astype(
            int
        )
        
        for i in range(self.cluster_num):
            class_idx = prediction == i
            if class_idx.sum() == 0:
                continue
            confidence_class = confidence[class_idx]
            num = min(confidence_class.shape[0], pseudo_per_class)
            confident_idx = torch.argsort(-confidence_class)
            for j in range(num):
                idx = tmp[class_idx][confident_idx[j]]
                pseudo_label_nxt[idx] = i

        todo_index = pseudo_label_cur == -1
        pseudo_label_cur[todo_index] = pseudo_label_nxt[todo_index]
        pseudo_label_nxt = pseudo_label_cur
        pseudo_label_nxt[unconfident_pred_index] = -1
        return pseudo_label_nxt.cpu(), index

    def forward(self, z, pseudo_label):
        n = z.shape[0]
        assert n % self.multiplier == 0

        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            pseudo_label_list = [
                torch.zeros_like(pseudo_label) for _ in range(dist.get_world_size())
            ]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            z_list = diffdist.functional.all_gather(z_list, z)
            pseudo_label_list = diffdist.functional.all_gather(
                pseudo_label_list, pseudo_label
            )
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            pseudo_label_list = [
                chunk for x in pseudo_label_list for chunk in x.chunk(self.multiplier)
            ]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            pesudo_label_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.multiplier + m])
                    pesudo_label_sorted.append(
                        pseudo_label_list[i * self.multiplier + m]
                    )
            z_i = torch.cat(
                z_sorted[: int(self.multiplier * dist.get_world_size() / 2)], dim=0
            )
            z_j = torch.cat(
                z_sorted[int(self.multiplier * dist.get_world_size() / 2) :], dim=0
            )
            pseudo_label = torch.cat(pesudo_label_sorted, dim=0,)
            n = z_i.shape[0]

        else:
            z_list = [z]  # 在非分布式环境下只有一个列表
            pseudo_label_list = [pseudo_label]  # 同样，pseudo_label 也只有一个列表

            # 直接进行拆分和合并，不再使用分布式的 all_gather
            z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
            pseudo_label_list = [chunk for x in pseudo_label_list for chunk in x.chunk(self.multiplier)]

            # 直接进行排序，不再需要分布式环境下的排序
            z_sorted = []
            pseudo_label_sorted = []
            for m in range(self.multiplier):
                z_sorted.append(z_list[m])
                pseudo_label_sorted.append(pseudo_label_list[m])

            # 将数据分为 z_i 和 z_j，这里不需要分布式合并，只是在本地进行分割
            z_i = torch.cat(z_sorted[:int(self.multiplier / 2)], dim=0)
            z_j = torch.cat(z_sorted[int(self.multiplier / 2):], dim=0)
            pseudo_label = torch.cat(pseudo_label_sorted, dim=0)
            
            n = z_i.shape[0]



        invalid_index = pseudo_label == -1
        mask = torch.eq(pseudo_label.view(-1, 1), pseudo_label.view(1, -1)).cuda()
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(n).float().cuda()
        mask &= ~(mask_eye.bool())
        mask = mask.float()

        contrast_count = self.multiplier
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.tau
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask_with_eye = mask | mask_eye.bool()
        # mask = torch.cat(mask)
        mask = mask.repeat(anchor_count, contrast_count)
        mask_eye = mask_eye.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(n * anchor_count).view(-1, 1).cuda(),
            0,
        )
        logits_mask *= 1 - mask
        mask_eye = mask_eye * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_eye * log_prob).sum(1) / mask_eye.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, n).mean()

        return instance_loss

class ClusterLossBoost(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    LARGE_NUMBER = 1e4

    def __init__(self, multiplier=1, distributed=False, cluster_num=10):
        super().__init__()
        self.multiplier = multiplier
        self.distributed = distributed
        self.cluster_num = cluster_num

    def forward(self, c, pseudo_label):
        if self.distributed:
            pesudo_label_list = [
                torch.zeros_like(pseudo_label) for _ in range(dist.get_world_size())
            ]
            pesudo_label_list = diffdist.functional.all_gather(
                pesudo_label_list, pseudo_label
            )
            pesudo_label_list = [
                chunk for x in pesudo_label_list for chunk in x.chunk(self.multiplier)
            ]
            pesudo_label_sorted = []
            for m in range(self.multiplier):
                for i in range(dist.get_world_size()):
                    pesudo_label_sorted.append(
                        pesudo_label_list[i * self.multiplier + m]
                    )
            pesudo_label_all = torch.cat(pesudo_label_sorted, dim=0)
            
        else:
            # 先将 pseudo_label_list 和 c_list 直接处理成一个列表
            pesudo_label_list = [pseudo_label]

            # 如果有多个 augmenter 或者增广的样本，我们直接合并
            pesudo_label_list = [
                chunk for x in pesudo_label_list for chunk in x.chunk(self.multiplier)
            ]
            
            # 直接在本地进行排序，不再分布式环境下进行
            pesudo_label_sorted = []
            for m in range(self.multiplier):
                pesudo_label_sorted.append(pesudo_label_list[m])

            # 连接所有的 pseudo_labels
            pesudo_label_all = torch.cat(pesudo_label_sorted, dim=0)
    
    
        pseudo_index = pesudo_label_all != -1
        pesudo_label_all = pesudo_label_all[pseudo_index]
        idx, counts = torch.unique(pesudo_label_all, return_counts=True)
        freq = pesudo_label_all.shape[0] / counts.float()
        weight = torch.ones(self.cluster_num).cuda()
        weight[idx] = freq
        pseudo_index = pseudo_label != -1
        if pseudo_index.sum() > 0:
            criterion = nn.CrossEntropyLoss(weight=weight).cuda()
            loss_ce = criterion(
                c[pseudo_index], pseudo_label[pseudo_index].cuda()
            )
        else:
            loss_ce = torch.tensor(0.0, requires_grad=True).cuda()
            
        return loss_ce

EPS = 10 ** -6

class DivClustLoss(torch.nn.Module):

    def __init__(self, threshold=1., NMI_target=1., NMI_interval=5, threshold_rate=0.99, divclust_mbank_size=10000, *args, **kwargs):
        super(DivClustLoss, self).__init__()
        self.threshold = threshold
        self.NMI_target = NMI_target
        self.NMI_interval = NMI_interval
        self.threshold_rate = threshold_rate
        self.current_threshold = threshold
        self.divclust_mbank_size = divclust_mbank_size
        self.memory_bank = None

    def loss(self, assignments, threshold):
        if not isinstance(assignments, torch.Tensor):
            assignments = torch.stack(assignments)
        K, N, C = assignments.shape
        id_rem = F.one_hot(torch.arange(K, device=assignments.device), K).bool()
        clustering_similarities = torch.einsum("qbc,kbd->qkcd", assignments, assignments).permute(1, 0, 2, 3)[
            ~id_rem].view(K * (K - 1), C, C)

        clustering_sim_aggr = clustering_similarities.max(-1)[0].mean(-1)
        loss = F.relu(clustering_sim_aggr - threshold).sum()

        return loss

    def forward(self, assignments: torch.Tensor, step=None):
        if isinstance(assignments, torch.Tensor):
            if len(assignments.shape) == 2:
                assignments = assignments.unsqueeze(0)
        clusterings = len(assignments)

        if clusterings == 1 or self.NMI_target == 1:
            return torch.tensor(0., device=assignments.device, requires_grad=True), self.threshold, assignments

        if self.NMI_target == 1:
            threshold = self.get_adaptive_threshold(threshold, self.adaptive_threshold, step)
        else:
            self.update_mb(assignments)
            threshold = self.get_NMI_threshold(self.NMI_target, step)
        self.current_threshold = threshold

        if isinstance(assignments, torch.Tensor):
            assignmentsl2 = F.normalize(assignments, p=2, dim=1)
        else:
            assignmentsl2 = [F.normalize(assignments_k, p=2, dim=0) for assignments_k in assignments]

        if threshold == 1.:
            return torch.tensor(0., device=assignments.device, requires_grad=True), threshold, assignments

        loss = self.loss(assignmentsl2, self.current_threshold)
        return loss, threshold, assignments

    @torch.no_grad()
    def update_mb(self, assignments):
        labels = assignments.argmax(-1)
        if self.memory_bank is None:
            self.memory_bank = labels.cpu().numpy()
        else:
            self.memory_bank = np.concatenate([labels.cpu().numpy(), self.memory_bank], axis=1)
        self.memory_bank = self.memory_bank[:, :self.divclust_mbank_size]

    def get_NMI_threshold(self, NMI_target, step):
        threshold = self.current_threshold
        if step is None or step % self.NMI_interval == 0:
            k = self.memory_bank.shape[0]
            NMIs = []
            for k1 in range(k):
                for k2 in range(k1 + 1, k):
                    NMIs.append(normalized_mutual_info_score(self.memory_bank[k1], self.memory_bank[k2]))
            NMI = np.mean(NMIs)
            if NMI > NMI_target:
                threshold = self.current_threshold * self.threshold_rate
            else:
                threshold = self.current_threshold * (2-self.threshold_rate)
            threshold = max(0, threshold)
            threshold = min(1., threshold)
        return threshold


class CCLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(CCLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.temperature = temperature

        self.bs = None
        self.batch_indexes = None
        self.positive_mask = None
        self.negative_mask = None
        self.labels = None

        self.cluster_labels = None
        self.cluster_negative_mask = None
        self.cluster_positive_mask = None
        self.cluster_indexes = None

    def instance_loss(self, z1, z2):
        bs = z1.shape[0]
        if self.batch_indexes is None or self.bs != bs:
            self.bs = bs
            self.batch_indexes = torch.arange(bs, device=z1.device)
            self.positive_mask = F.one_hot(torch.cat([self.batch_indexes + bs, self.batch_indexes]), bs * 2).bool()
            self.negative_mask = (1 - F.one_hot(torch.cat([self.batch_indexes, self.batch_indexes + bs]), bs * 2)-self.positive_mask.float()).bool()
            self.labels = torch.zeros((2*bs,),device=z1.device).long()

        z = F.normalize(torch.cat([z1, z2], dim=0),p=2,dim=-1)
        s = z @ z.T
        positives = s[self.positive_mask].view(2 * bs, 1)
        negatives = s[self.negative_mask].view(2 * bs, -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        loss = self.CE(logits, self.labels)
        return loss

    def cluster_loss(self, p1, p2):
        if len(p1.shape)==2:
            p1 = p1.unsqueeze(0)
        if len(p2.shape)==2:
            p2 = p2.unsqueeze(0)
        k,bs,c = p1.shape
        if self.cluster_indexes is None or self.bs != bs:
            self.bs = bs
            self.cluster_indexes = torch.arange(c, device=p1.device)
            self.cluster_positive_mask = torch.stack(k*[F.one_hot(torch.cat([self.cluster_indexes + c, self.cluster_indexes]), c * 2).bool()])
            negative_mask = torch.stack(k*[F.one_hot(torch.cat([self.cluster_indexes, self.cluster_indexes + c]), c * 2)])
            self.cluster_negative_mask = (1 - negative_mask - self.cluster_positive_mask.float()).bool()
            self.cluster_labels = torch.zeros((2*c,),device=p1.device).long()

        p = torch.cat([p1, p2], dim=2)
        if len(p.shape)==2:
            p = p.unsqueeze(0)
        p = F.normalize(p,dim=1)
        s = torch.einsum("kna, knb->kab", p, p)
        positives = s[self.cluster_positive_mask].view(k,2*c,-1)
        negatives = s[self.cluster_negative_mask].view(k,2*c,-1)
        logits = torch.cat([positives, negatives], dim=2)

        loss_ce = []
        loss_ne = []
        for k_ in range(k):
            loss_ce.append(self.CE(logits[k_], self.cluster_labels))
            p_i = p1[k_].sum(0).view(-1)
            p_i /= p_i.sum()
            ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
            p_j = p2[k_].sum(0).view(-1)
            p_j /= p_j.sum()
            ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
            loss_ne.append(ne_i + ne_j)
        return loss_ce, loss_ne

    def forward(self, p1, p2, z1,z2):
        loss_ce, loss_ne = self.cluster_loss(p1,p2)
        loss_cc = self.instance_loss(z1, z2)
        return loss_ce, loss_ne, loss_cc

# SCAN
class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        if anchors_weak[0].sum()>=0.99999 and anchors_weak[0].sum()<=1.00001:
            weak_anchors_prob = anchors_weak

        else:
            weak_anchors_prob = self.softmax(anchors_weak)

        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts=True)
            freq = 1 / (counts.float() / n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight=weight, reduction='mean')
        # print(n)
        return loss
    
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)
    
class SCANLoss(nn.Module):
    def __init__(self, entropy_weight=2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss
    
def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = 1e-8)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))