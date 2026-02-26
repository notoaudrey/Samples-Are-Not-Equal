'''
@File  :main_cdc_sample.py
@Date  :2025/8/27 14:44
@Desc  :
'''

import logging
import re
import warnings
import wandb
from torch.utils.data import DataLoader, Subset
import pickle

logging.captureWarnings(True)
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.'.format(re.escape(__name__)))
warnings.warn("This is a DeprecationWarning",category=DeprecationWarning)

import argparse
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from cdc.args import parse_cfg, get_model, get_strong_transformations,\
    get_val_transformations, get_standard_transformations,\
    get_train_dataloader, get_val_dataloader,\
    get_train_dataset,get_val_dataset, get_optimizer
from cdc.utils.evaluate_utils import get_predictions, \
    hungarian_evaluate, calibration_evaluate,  hungarian_evaluate_hard
from cdc.methods.calibrate_train import initialize_weights,  initialize_weights_bias
from cdc.backbones.models import CaliMLP
from cdc.methods.dyn_train import SampleMasterTracker, train_cali_sample_speed

FLAGS = argparse.ArgumentParser(description='CDC Model')
FLAGS.add_argument('--config_env', default='scripts/cdc/env.yaml', help='Location of path config file')
FLAGS.add_argument('--config_exp', default='scripts/cdc/stl/cdc_ini_bias_a20_sample_stabilityloss_t1_seed5.yaml', help='Location of experiments config file')

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
    val_dataloader = get_val_dataloader(cfg, val_dataset)
    
    print('Strong transforms:', strong_transformations)
    print('Standard transforms:', standard_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    
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
    k = cfg.get('k', 10)
    initialize_weights_bias(cfg, model, cali_mlp, features, val_dataloader, alpha=alpha,target_class=target_class) 

    predictions, features = get_predictions(cfg, val_dataloader, model, return_features=True)
    clustering_stats_bias, indices = hungarian_evaluate_hard(cfg, cfg['cdc_dir'], 0, 0,
                                        predictions, title=cfg['cluster_eval']['plot_title'],
                                        compute_confusion_matrix=False, easy=True)
    print('CDC-Clu-ini-bias ', clustering_stats_bias)

    predictions = get_predictions(cfg, val_dataloader, model, cali_mlp = cali_mlp)
    clustering_stats = calibration_evaluate(cfg, cfg['cdc_dir'], 0, 0, predictions,
                                                title=cfg['cluster_eval']['plot_title'],
                                                compute_confusion_matrix=False)
    print('CDC-Cal ', clustering_stats)

    torch.save({'optimizer_clu': optimizer_clu.state_dict(),
                    'optimizer_cali': optimizer_cali.state_dict(),
                    'model': model.state_dict(),
                    'cali_mlp': cali_mlp.state_dict(),
                    'epoch': 0},
                    os.path.join(cfg['cdc_dir'], "checkpoint_highinin.pth.tar"))

    log_path = os.path.join(cfg['cdc_dir'], 'training_log.log')
    log_file = open(log_path, 'a')
    log_file.write(f'CDC-Clu-ini: {clustering_stats}\n')
    log_file.write(f'CDC-Clu-ini-bias: {clustering_stats_bias}\n')

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
    
    thresh=cfg.get('t', 0.01)
    window = cfg.get('w', 3)
    shake = cfg.get('shake', 1.0)
    shake_epoch = cfg.get('shake_epoch', -1)
    s= cfg.get('s', 1.0)
    tracker = SampleMasterTracker(cfg, num_samples=len(train_dataloader.dataset),
                              delta_thresh=thresh, window=window, shake_thresh=shake, shake_epoch=shake_epoch, s=s)
    

    metrics_log = {"epoch": [], "acc": [], "remove_acc": [], "highconf_acc": [], "highconf_acc_balanced": []}
    epoch_time_list = []
    for epoch in range(start_epoch, cfg['max_epochs']):
        print('Epoch %d/%d' % (epoch + 1, cfg['max_epochs']))
        # Train
        print('Train ...')
        #train_cali(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_clu, epoch, start_epoch)
        epoch_time=train_cali_sample_speed(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_clu, tracker, stabilityloss=True)
        
        epoch_time_list.append(epoch_time)
        print("epoch_time: ", epoch_time)

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
                
        #log_file.write(f'Train samples: {len(train_dataloader.dataset)}\n')
        #log_file.write(f'Val samples: {len(val_dataloader.dataset)}\n\n')

        log_file.close()    
            
        if epoch == prun_epoch:
            torch.save({'optimizer_clu': optimizer_clu.state_dict(),
                    'optimizer_cali': optimizer_cali.state_dict(),
                    'model': model.state_dict(),
                    'cali_mlp': cali_mlp.state_dict(),
                    'epoch': epoch},
                   os.path.join(cfg['cdc_dir'],f"checkpoint_{prun_epoch}.pth.tar"))
            print('prun sample ...')
            predictions, features = get_predictions(cfg, val_dataloader, model, return_features=True)
            clustering_stats, indices = hungarian_evaluate_hard(cfg, cfg['cdc_dir'], epoch, 0, predictions,
                                                title=cfg['cluster_eval']['plot_title'],
                                                compute_confusion_matrix=False, easy=True)
            predictions = get_predictions(cfg, val_dataloader, model, cali_mlp = cali_mlp)
            clustering_stats, indices = calibration_evaluate(cfg, cfg['cdc_dir'], epoch, 0, predictions,
                                                title=cfg['cluster_eval']['plot_title'],
                                                compute_confusion_matrix=False, features=features, flag=True)
            
            total_indices = list(set(indices))
            subset = Subset(train_dataset, total_indices)
            train_dataloader = get_train_dataloader(cfg, subset)
            
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
            
            
    np.save("shake_delta2_history.npy", np.array(tracker.shake_delta2_history, dtype=object))
    
        
    # Evaluate and save the final model
    print('Evaluate best model at the end')
    
    checkpoint = torch.load(cfg['cdc_best_model'], map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    cali_mlp.load_state_dict(checkpoint['cali_mlp'], strict=False)
    
    predictions, features = get_predictions(cfg, val_dataloader, model, return_features=True)
    # clustering_stats = hungarian_evaluate(cfg, cfg['cdc_dir'], cfg['max_epochs'], 0, predictions,
    #                                           title=cfg['cluster_eval']['plot_title'],
    #                         class_names=val_dataloader.dataset.classes,
    #                         compute_confusion_matrix=True,
    #                         confusion_matrix_file=os.path.join(cfg['cdc_dir'], 'confusion_matrix.png'), save_wrong=True,bins=indices_per_bin)  
    clustering_stats_bias, indices = hungarian_evaluate_hard(cfg, cfg['cdc_dir'], cfg['max_epochs'], 0,
                                        predictions, title=cfg['cluster_eval']['plot_title'],
                                        compute_confusion_matrix=False, easy=True)
    print(clustering_stats_bias)

    predictions = get_predictions(cfg, val_dataloader, model, cali_mlp = cali_mlp)
    clustering_stats2 = calibration_evaluate(cfg, cfg['cdc_dir'], 999, 0, predictions,
                                                title=cfg['cluster_eval']['plot_title'],
                                                compute_confusion_matrix=False, remove = tracker.removed, shake = tracker.shake)
    print(clustering_stats2)
    
    log_file = open(log_path, 'a')
    log_file.write(f'best EVA: {clustering_stats}\n{clustering_stats2}')
    log_file.close()

    """ import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_log["epoch"], metrics_log["acc"], label="Overall acc", marker="o")
    plt.plot(metrics_log["epoch"], metrics_log["remove_acc"], label="Remove acc", marker="x")
    plt.plot(metrics_log["epoch"], metrics_log["highconf_acc"], label="High-conf acc", marker="s")
    plt.plot(metrics_log["epoch"], metrics_log["highconf_acc_balanced"], label="High-conf balanced acc", marker="d")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True)

    plt.ylim(60, 100)

    plt.tight_layout()
    
    plt.savefig(os.path.join(cfg['cdc_dir'], "acc-t.png"))
    
    
    tracker.plot_delta2_distributions(bins=100, interval=10)
    tracker.plot_delta2_trend()

    # ====== 训练结束后，展示趋势和均值 ======
    import matplotlib.pyplot as plt
    mean_time = np.mean(epoch_time_list)
    print(f"\n平均每个 epoch 耗时: {mean_time:.2f} 秒")

    plt.figure(figsize=(8,4))
    plt.plot(range(1, len(epoch_time_list)+1), epoch_time_list, marker='o', label='Epoch Time')
    plt.axhline(mean_time, color='red', linestyle='--', label=f'Mean = {mean_time:.2f}s')
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Training Time per Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(cfg['cdc_dir'], "epoch-time.png"))
     """

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