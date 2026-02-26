"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        from faiss import normalize_L2
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]

        normalize_L2(features)

        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)

        index.train(features)

        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            if topk!=0:
                neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
                anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
            else:
                neighbor_targets = np.take(targets, indices[:,0:], axis=0)
                anchor_targets = np.repeat(targets.reshape(-1, 1), 1, axis=1)

            accuracy = np.mean(neighbor_targets == anchor_targets)
            # try:
            #     print(anchor_targets.max())
            # except:
            #     pass
            return distances, indices, accuracy
        
        else:
            return distances, indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        # print(b,features.detach().shape,self.features[self.ptr:self.ptr+b].shape)
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images, targets = batch[0], batch[1]
        output = model(images.cuda(non_blocking=True), forward_pass = 'backbone')
        # print(output)
        if isinstance(output, list):
            output=output[0]
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))
