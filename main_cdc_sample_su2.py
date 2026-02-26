'''
@File  :main_cdc_sample_su.py
@Date  :2025/8/27 14:44
@Desc  :
'''

import logging
import re
import warnings
import wandb
from torch.utils.data import DataLoader, Subset
import pandas as pd  

logging.captureWarnings(True)
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.'.format(re.escape(__name__)))
warnings.warn("This is a DeprecationWarning",category=DeprecationWarning)

import argparse
import os
import torch
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import tracemalloc
import psutil
import matplotlib.pyplot as plt

from cdc.args import parse_cfg, get_model, get_strong_transformations,\
    get_val_transformations, get_standard_transformations,\
    get_train_dataloader, get_val_dataloader,\
    get_train_dataset,get_val_dataset, get_optimizer
from cdc.utils.evaluate_utils import get_predictions, \
    hungarian_evaluate, calibration_evaluate,  hungarian_evaluate_hard, get_predictions_su
from cdc.methods.calibrate_train import initialize_weights, initialize_weights_bias, train_cali
from cdc.backbones.models import CaliMLP
from cdc.methods.dyn_train import SampleMasterTracker, train_cali_sample_su


FLAGS = argparse.ArgumentParser(description='CDC Model')
FLAGS.add_argument('--config_env', default='scripts/cdc/env.yaml', help='Location of path config file')
FLAGS.add_argument('--config_exp', default='scripts/cdc/cifar10/cdc_res18_a10_t1_seed5.yaml', help='Location of experiments config file')

process = psutil.Process(os.getpid())

def ram(msg):
    print(f"[RAM] {msg}: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def snap(msg):
    current, peak = tracemalloc.get_traced_memory()
    print(f"[PythonMem] {msg}: current={current/1024:.2f} KB  peak={peak/1024:.2f} KB")

from torchvision.datasets import STL10
from torchvision import transforms
import matplotlib.pyplot as plt

def denormalize(img_tensor, mean, std):
    # img_tensor: [C,H,W]
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img_tensor * std + mean


def visualize_indices(low_idx, high_idx, dataset,save_dir="./delta2_vis"):
    os.makedirs(save_dir, exist_ok=True)

    # --- 加载 STL10 数据（train split, 不带 transform）---
    # dataset = STL10(root="./data", split='train', download=True,
    #                 transform=transforms.ToTensor())

    # ---- 展示函数 ----
    mean = [0.44671062, 0.43980984, 0.40664645]
    std  = [0.22414587, 0.22148906, 0.22389975]

    low_dir  = os.path.join(save_dir, "low")
    high_dir = os.path.join(save_dir, "high")
    os.makedirs(low_dir, exist_ok=True)
    os.makedirs(high_dir, exist_ok=True)

    def save_grid(indices, filename):
        imgs = []
        for idx in indices:
            img, _ , _= dataset[idx]  # img: [3,H,W]


            img = denormalize(img, mean, std)
            img = img.permute(1, 2, 0).numpy()

            # 限制展示范围，避免警告
            img = np.clip(img, 0, 1)
            imgs.append(img)

        # 一行展示所有样本
        fig, axes = plt.subplots(5, 10, figsize=(20, 10))
        axes = axes.flatten()

        for ax, img in zip(axes, imgs):
            ax.imshow(img)
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=200)
        plt.close()

    def save_one(img_tensor, save_path):
        img = denormalize(img_tensor, mean, std)
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        plt.figure(figsize=(2,2))
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', pad_inches=0)
        plt.close()

    # ---- 保存两组图片 ----
    # save_grid(low_idx,  "low_delta2_top50.png")
    # save_grid(high_idx, "high_delta2_top50.png")

     # -------- save 50 low --------
    for i, idx in enumerate(low_idx):
        img, _, _ = dataset[idx]
        save_path = os.path.join(low_dir, f"low_{i:03d}_idx_{idx}.png")
        save_one(img, save_path)

    # -------- save 50 high --------
    for i, idx in enumerate(high_idx):
        img, _, _ = dataset[idx]
        save_path = os.path.join(high_dir, f"high_{i:03d}_idx_{idx}.png")
        save_one(img, save_path)

    print("Done: all images saved.")

    # print(f"Saved to: {save_dir}/low_delta2_top50.png and high_delta2_top50.png")


def main():
    args = FLAGS.parse_args()
    cfg = parse_cfg(args.config_env, args.config_exp)
    print(cfg)
    # Data
    print('Get dataset and dataloaders')
    strong_transformations = get_strong_transformations(cfg)
    standard_transformations = get_standard_transformations(cfg)
    val_transformations = get_val_transformations(cfg)

    train_dataset = get_train_dataset(cfg, {'val': val_transformations,
                                            'standard': standard_transformations,
                                            'augment': strong_transformations},
                                        split=cfg['data']['split'], augmented = True)
    val_dataset = get_val_dataset(cfg, val_transformations)
    train_dataloader = get_train_dataloader(cfg, train_dataset)
    sample_dataloader =  get_val_dataloader(cfg, train_dataset)
    val_dataloader = get_val_dataloader(cfg, val_dataset)

    sample_number = 50000
    indices_k = list(range(sample_number))
    train_dataset = torch.utils.data.Subset(train_dataset, indices_k)
    val_dataset = torch.utils.data.Subset(val_dataset, indices_k)
    train_dataloader = get_train_dataloader(cfg, train_dataset)
    val_dataloader = get_val_dataloader(cfg, val_dataset)
    
    print('Strong transforms:', strong_transformations)
    print('Standard transforms:', standard_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    
    prun_epoch = cfg.get('prun_epoch', -1)
    print('Pruning epoch:', prun_epoch)
    
    # Model
    print('Get model')
    model = get_model(cfg, cfg['pretext']['enable'])

    cali_mlp = CaliMLP(cfg)
    cali_mlp = torch.nn.DataParallel(cali_mlp)
    cali_mlp = cali_mlp.cuda()

    # Optimizer
    print('Get optimizer')
    optimizer_clu = get_optimizer(cfg, model)
    optimizer_cali = torch.optim.Adam(cali_mlp.parameters(), lr=cfg['optimizer']['lr'],
                                      **cfg['optimizer']['kwargs'])
    # wandb
    wandb.watch(model, log="all")
        
    # Evaluate
    predictions, features = get_predictions(cfg, val_dataloader, model, return_features=True)
    clustering_stats, indices = hungarian_evaluate_hard(cfg, cfg['cdc_dir'], 0, 0,
                                        predictions, title=cfg['cluster_eval']['plot_title'],
                                        compute_confusion_matrix=False)
    print('CDC-Clu ', clustering_stats)

    # Initialize weights
    initialize_weights(cfg, model, cali_mlp, features, val_dataloader)    
    predictions, features = get_predictions(cfg, val_dataloader, model, return_features=True)
    clustering_stats, indices = hungarian_evaluate_hard(cfg, cfg['cdc_dir'], 0, 0,
                                        predictions, title=cfg['cluster_eval']['plot_title'],
                                        compute_confusion_matrix=False, easy=True)
    print('CDC-Clu-ini ', clustering_stats)

    # Initialize weights bias
    alpha= cfg.get('alpha', 1.0)
    initialize_weights_bias(cfg, model, cali_mlp, features, val_dataloader, k=10, alpha=alpha)    
    predictions, features = get_predictions(cfg, val_dataloader, model, return_features=True)
    clustering_stats_bias, indices = hungarian_evaluate_hard(cfg, cfg['cdc_dir'], 0, 0,
                                        predictions, title=cfg['cluster_eval']['plot_title'],
                                        compute_confusion_matrix=False, easy=True)
    print('CDC-Clu-ini-bias ', clustering_stats_bias)

    torch.save({'optimizer_clu': optimizer_clu.state_dict(),
                    'optimizer_cali': optimizer_cali.state_dict(),
                    'model': model.state_dict(),
                    'cali_mlp': cali_mlp.state_dict(),
                    'epoch': 0},
                    os.path.join(cfg['cdc_dir'], "checkpoint_highinin.pth.tar"))


    log_path = os.path.join(cfg['cdc_dir'], 'training_log.log')
    log_file = open(log_path, 'a')
    log_file.write(f'CDC-Clu-ini: {clustering_stats}\n')
    
    # Checkpoint
    if os.path.exists(cfg['cdc_checkpoint']):
        print('Restart from checkpoint {}'.format(cfg['cdc_checkpoint']))
        checkpoint = torch.load(cfg['cdc_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        cali_mlp.load_state_dict(checkpoint['cali_mlp'], strict=False)
        start_epoch = checkpoint['epoch']        
    else:
        print('No checkpoint file at {}'.format(cfg['cdc_checkpoint']))
        start_epoch = 0

    # Main loop
    print('Starting main loop', 'blue')
    best_acc = -1
    
    # tracemalloc.start()

    thresh=cfg.get('t',0.1)
    window = cfg.get('w', 3)
    shake = cfg.get('shake', 0.5)
    shake_epoch = cfg.get('shake_epoch', -1)
    s= cfg.get('s', 0.2)
    tracker = SampleMasterTracker(cfg, num_samples=len(train_dataloader.dataset),
                              delta_thresh=thresh, window=window, shake_thresh=shake, shake_epoch=shake_epoch, s=s)
    
    metrics_log = {"epoch": [], "acc": [], "remove_acc": [], "highconf_acc": [], "highconf_acc_balanced": []}


    ini_dataloader = get_train_dataloader(cfg, train_dataset)

    for epoch in range(start_epoch, cfg['max_epochs']):
        print('Epoch %d/%d' % (epoch + 1, cfg['max_epochs']))

        total_indices= set(range(len(train_dataset)))
        remove_indices = tracker.removed
        keep_indices= list(total_indices-remove_indices)
        subset = Subset(train_dataset, keep_indices)
        train_dataloader = get_train_dataloader(cfg, subset)
        print("len(train_dataloader): ", len(train_dataloader.dataset))

        start_time =  time.time()
        for step, batch in enumerate(ini_dataloader):
            images = batch['image'].cuda(non_blocking=True)
            images_augmented = batch['image_augmented'].cuda(non_blocking=True)
            model.train()
            cali_mlp.train()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    feature_weak = model(images, forward_pass='backbone')
                    feature_augmented = model(images_augmented, forward_pass='backbone')
                    images_index = batch['index'].cuda(non_blocking=True)
                    output_cali = cali_mlp(feature_weak, forward_pass='calibration')
            feature_stability = F.cosine_similarity(feature_weak, feature_augmented, dim=1)
            stability_loss = 1 - feature_stability
            cali_softmax = F.softmax(output_cali, dim=1)
            cali_prob, cali_label = torch.max(cali_softmax, dim=1)
            tracker.update(
                indices=images_index.tolist(),
                labels=cali_label.tolist(),
                losses=stability_loss.tolist()
            )
        
        tracker.step()

        # Train
        print('Train ...')
        train_cali(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_clu, epoch, start_epoch)
        
        # Evaluate
        log_file = open(log_path, 'a')
        log_file.write(f'Epoch {epoch+1} - Validation prediction\n')
        log_file.write(f'Train data removed - {len(tracker.removed)}/{len(train_dataloader)}\n')

        if (epoch+1) % 1 == 0:
            print('Make prediction on validation set ...')
            predictions = get_predictions(cfg, val_dataloader, model)
            clustering_stats, indices = hungarian_evaluate_hard(cfg, cfg['cdc_dir'], epoch, 0, predictions,
                                                title=cfg['cluster_eval']['plot_title'],
                                                compute_confusion_matrix=False, _indices=indices)
            print('CDC-Clu ', clustering_stats)
            log_file.write(f'CDC-Clu: {clustering_stats}\n')
            predictions = get_predictions(cfg, val_dataloader, model, cali_mlp = cali_mlp)
            clustering_stats = calibration_evaluate(cfg, cfg['cdc_dir'], epoch, 0, predictions,
                                                title=cfg['cluster_eval']['plot_title'],
                                                compute_confusion_matrix=False, remove = tracker.removed)
            print('CDC-Cal ', clustering_stats)
            log_file.write(f'CDC-Cal: {clustering_stats}\n\n') 

            if tracker.removed is not None and len(tracker.removed) > 0:
                metrics_log["epoch"].append(epoch)
                metrics_log["acc"].append(clustering_stats["ACC"])
                metrics_log["remove_acc"].append(clustering_stats["remove_acc"]*100)
                metrics_log["highconf_acc"].append(clustering_stats["highconf_acc"]*100)
                metrics_log["highconf_acc_balanced"].append(clustering_stats["highconf_acc_balanced"]*100)
                
        log_file.write(f'Train samples: {len(train_dataloader.dataset)}\n')
        log_file.write(f'Val samples: {len(val_dataloader.dataset)}\n\n')
        log_file.close()    
            
            
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer_clu': optimizer_clu.state_dict(),
                    'optimizer_cali': optimizer_cali.state_dict(),
                    'model': model.state_dict(),
                    'cali_mlp': cali_mlp.state_dict(),
                    'epoch': epoch + 1},
                   cfg['cdc_checkpoint'])
        if best_acc < clustering_stats['ACC']:
            torch.save({
                        'model': model.state_dict(),
                        'cali_mlp': cali_mlp.state_dict(),
                        'epoch': epoch + 1},
                       cfg['cdc_best_model'])
            best_acc = clustering_stats['ACC']
                    
    
    # Evaluate and save the final model
    print('Evaluate best model at the end')
    
    checkpoint = torch.load(cfg['cdc_best_model'], map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    cali_mlp.load_state_dict(checkpoint['cali_mlp'])
    
    predictions, features = get_predictions(cfg, val_dataloader, model, return_features=True)
    clustering_stats = hungarian_evaluate(cfg, cfg['cdc_dir'], cfg['max_epochs'], 0, predictions,
                                              title=cfg['cluster_eval']['plot_title'],
                            class_names=val_dataloader.dataset.classes,
                            compute_confusion_matrix=True,
                            confusion_matrix_file=os.path.join(cfg['cdc_dir'], 'confusion_matrix.png'), save_wrong=True)  
    print(clustering_stats)
    
    log_file = open(log_path, 'a')
    log_file.write(f'best EVA: {clustering_stats}\n')
    log_file.close()

    log_path = os.path.join(cfg['cdc_dir'], "metrics_log.csv")
    pd.DataFrame(metrics_log).to_csv(log_path, index=False)
    print(f"Saved metrics log to: {log_path}")

    
if __name__ == "__main__":
    seed = 5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(True)
    print('seed:', seed)
    main()
    wandb.finish()