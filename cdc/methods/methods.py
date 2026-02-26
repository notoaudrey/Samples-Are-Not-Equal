'''
@File  :methods.py
@Author:cjh
@Date  :2023/1/29 16:15
@Desc  :
'''
import torch
import torch.nn.functional as F

def fixed_train(clustering_stats, train_dataloader, model, criterion, optimizer, epoch):
    print()

def supervised_train(clustering_stats, train_dataloader, model, criterion, optimizer, epoch, num_classes=10):
    model.train()
    for i, batch in enumerate(train_dataloader):
        images = batch['image_augmented'].cuda(non_blocking=True)
        # images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        gt = batch['target'].cuda(non_blocking=True)
        #gt_hot = F.one_hot(gt, num_classes = num_classes).float()
        output = model(images)[0]
        
        #import pdb; pdb.set_trace()
        
        loss = criterion(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if len(clustering_stats) != 0:
            gt_map = clustering_stats['hungarian_match']
        for pre, post in gt_map:
            gt[batch['target'] == post] = pre

