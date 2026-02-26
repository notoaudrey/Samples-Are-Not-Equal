'''
@File  :args.py
@Date  :2023/1/29 16:15
@Desc  :
'''
import os

import wandb
import yaml
from easydict import EasyDict
import errno
import torch
import math
import numpy as np
import torchvision.transforms as transforms
from cdc.data.augment import Augment, Cutout
from cdc.data.collate import collate_custom
from torch.utils.data import ConcatDataset


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def parse_cfg(config_file_env, config_file_exp):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    wandb.init(project=cfg['project'], name=cfg['name'])

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['data']['dataset'])
    mkdir_if_missing(base_dir)
    
    if cfg['pretext']['enable']:
        pretext_dir = cfg['pretext']['dir']
        cfg['pretext_dir'] = pretext_dir
        cfg['pretext_model'] = os.path.join(pretext_dir, cfg['pretext']['ckpt'])
    if cfg['method'] == 'cdcv2':
        cdc_dir = os.path.join(base_dir, cfg['name'])
        mkdir_if_missing(cdc_dir)
        cfg['cdc_dir'] = cdc_dir
        cfg['cdc_checkpoint'] = os.path.join(cdc_dir, 'checkpoint.pth.tar')
        cfg['cdc_best_model'] = os.path.join(cdc_dir, 'best_model.pth.tar')
        cfg['cdc_model'] = os.path.join(cdc_dir, 'model.pth.tar')
    elif cfg['method'] == 'tcl':
        tcl_dir = os.path.join(base_dir, cfg['name'])
        mkdir_if_missing(tcl_dir)
        cfg['tcl_dir'] = tcl_dir
        cfg['tcl_checkpoint'] = os.path.join(tcl_dir, 'checkpoint.pth.tar')
        cfg['tcl_best_model'] = os.path.join(tcl_dir, 'best_model.pth.tar')
        cfg['tcl_model'] = os.path.join(tcl_dir, 'model.pth.tar')
    elif cfg['method'] == 'scan':
        scan_dir = os.path.join(base_dir, cfg['name'])
        mkdir_if_missing(scan_dir)
        cfg['scan_dir'] = scan_dir
        cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.pth.tar')
        cfg['scan_best_model'] = os.path.join(scan_dir, 'best_model.pth.tar')
        cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')

        scan_selflabel_dir = os.path.join(base_dir, cfg['name'])
        mkdir_if_missing(scan_selflabel_dir)
        cfg['scan_selflabel_dir'] = scan_selflabel_dir
        cfg['scan_selflabel_checkpoint'] = os.path.join(scan_selflabel_dir, 'checkpoint.pth.tar')
        cfg['scan_selflabel_best_model'] = os.path.join(scan_selflabel_dir, 'best_model.pth.tar')
        cfg['scan_selflabel_model'] = os.path.join(scan_selflabel_dir, 'model.pth.tar')

        cfg['topk_neighbors_train_path'] = os.path.join(scan_dir, 'topk-train-neighbors.npy')
        cfg['topk_neighbors_train_dist'] = os.path.join(scan_dir, 'topk-train-neighbors-dist.npy')
    
    return cfg

def get_strong_transformations(cfg):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(cfg['augmentation_strong']['crop_size']),
        Augment(cfg['augmentation_strong']['num_strong_augs']),
        transforms.ToTensor(),
        transforms.Normalize(**cfg['augmentation_strong']['normalize']),
        Cutout(
            n_holes=cfg['augmentation_strong']['cutout_kwargs']['n_holes'],
            length=cfg['augmentation_strong']['cutout_kwargs']['length'],
            random=cfg['augmentation_strong']['cutout_kwargs']['random'])])
    
def get_standard_transformations(cfg):
    return transforms.Compose([
        transforms.RandomResizedCrop(**cfg['augmentation_stantard']['random_resized_crop']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**cfg['augmentation_stantard']['normalize'])
    ])

def get_val_transformations(cfg):
    if cfg['data']['dataset'] in ["imagenetdogs", "imagenet10", "visda"]:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(cfg['augmentation_val']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(**cfg['augmentation_val']['normalize'])])
    else:
        return transforms.Compose([
            transforms.CenterCrop(cfg['augmentation_val']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(**cfg['augmentation_val']['normalize'])])

def get_train_dataset(cfg, transform, augmented = False, split=None, osr=False, known=None, to_neighbors_dataset = False, class_list=None):
    # Base dataset
    if cfg['data']['dataset'] == 'cifar10':
        if osr:
            from cdc.data.cifar20_dataset import CIFAR10, CIFAR10_OSR
            dataset = CIFAR10_OSR(known, cfg['data']['train_path'],
                          train=split, transform=transform, download=True).trainset
        else: 
            from cdc.data.cifar20_dataset import CIFAR10
            dataset = CIFAR10(cfg['data']['train_path'],
                            train=split, transform=transform, download=True)
            
        if class_list is not None:
            targets = dataset.targets if hasattr(dataset, 'targets') else dataset.labels
            if isinstance(targets, list):
                targets = np.array(targets)
            indices = [i for i, label in enumerate(targets) if label in class_list]
            dataset.data = dataset.data[indices]
            targets = targets[indices]
            label_map = {orig_label: new_label for new_label, orig_label in enumerate(class_list)}
            remapped_targets = np.array([label_map[label] for label in targets])

            # 更新到 dataset
            dataset.targets = torch.tensor(remapped_targets)

    elif cfg['data']['dataset'] == 'cifar20':
        from cdc.data.cifar20_dataset import CIFAR20
        dataset = CIFAR20(cfg['data']['train_path'],
                          train=split, transform=transform, download=True)
        
    elif cfg['data']['dataset'] == 'cmnist':
        from cdc.data.cmnist_dataset import ColorMNISTDataset
        dataset = ColorMNISTDataset(cfg['data']['train_path'],
                           transform=transform)

    elif cfg['data']['dataset'] == 'stl10':
        from cdc.data.stl10_dataset import STL10
        dataset = STL10(cfg['data']['train_path'],
                        split=split, transform=transform, download=True)

    elif cfg['data']['dataset'] in ["imagenet", "imagenet100", "tinyimagenet", "imagenetdogs", "imagenet10", "visda", "Office-31", "cub"]:
        from torchvision.datasets import ImageFolder
        #print(cfg['data']['train_path'])
        dataset = ImageFolder(cfg['data']['train_path'],
                              transform=transform)
        """ if cfg['data']['dataset'] == "cub"  and split == "train+test":
            dataset2 = ImageFolder(cfg['data']['val_path'],
                              transform=transform)
            dataset = ConcatDataset([dataset, dataset2]) """
        
    elif cfg['data']['dataset'] in ["waterbirds"]:
        from cdc.data.waterbirds_dataset import WaterbirdsDataset
        csv_file = os.path.join(cfg['data']['train_path'], "metadata.csv")
        dataset = WaterbirdsDataset(csv_file, cfg['data']['train_path'], 0, transform)
        
    elif cfg['data']['dataset'] in ["bar"]:
        from cdc.data.bar_dataset import BAR
        dataset = BAR(cfg['data']['train_path'], train=True, transform=transform)
    
    else:
        raise ValueError('Invalid train dataset {}'.format(cfg['data']['dataset']))
    # Wrap into other dataset (__getitem__ changes)
    if augmented:  # Dataset returns an image and an augmentation of that image.
        from cdc.data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)
        
    if to_neighbors_dataset: # Dataset returns an image and one of its nearest neighbors.
        from cdc.data.custom_dataset import NeighborsDataset
        indices = np.load(cfg['topk_neighbors_train_path'])
        dataset = NeighborsDataset(dataset, indices, cfg['data']['num_neighbors']) # Only use 5
        
    return dataset

def get_val_dataset(cfg, transform=None, class_list=None):
    # Base dataset
    if cfg['data']['dataset'] == 'cifar10':
        from cdc.data.cifar20_dataset import CIFAR10
        dataset = CIFAR10(cfg['data']['val_path'],
                          train='train+test', transform=transform, download=True)
        
        if class_list is not None:
            targets = dataset.targets 
            if isinstance(targets, list):
                targets = np.array(targets)
            indices = [i for i, label in enumerate(targets) if label in class_list]
            dataset.data = dataset.data[indices]

            targets = targets[indices]
            label_map = {orig_label: new_label for new_label, orig_label in enumerate(class_list)}
            remapped_targets = np.array([label_map[label] for label in targets])

            # 更新到 dataset
            dataset.targets = torch.tensor(remapped_targets)


    elif cfg['data']['dataset'] == 'cifar20':
        from cdc.data.cifar20_dataset import CIFAR20
        dataset = CIFAR20(cfg['data']['val_path'],
                          train='train+test', transform=transform, download=True)
        
    elif cfg['data']['dataset'] == 'cmnist':
        from cdc.data.cmnist_dataset import ColorMNISTDataset
        dataset = ColorMNISTDataset(cfg['data']['val_path'],
                           transform=transform)

    elif cfg['data']['dataset'] == 'stl10':
        from cdc.data.stl10_dataset import STL10
        dataset = STL10(cfg['data']['val_path'],
                        split='train+test', transform=transform, download=True)

    elif cfg['data']['dataset'] in ["imagenet", "imagenet100", "tinyimagenet", "imagenetdogs", "imagenet10", "visda", "Office-31", "cub"]:
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(cfg['data']['val_path'],
                              transform=transform)
        
    elif cfg['data']['dataset'] in ["waterbirds"]:
        from cdc.data.waterbirds_dataset import WaterbirdsDataset
        csv_file = os.path.join(cfg['data']['val_path'], "metadata.csv")
        dataset = WaterbirdsDataset(csv_file, cfg['data']['val_path'], 1, transform)
        
    elif cfg['data']['dataset'] in ["bar"]:
        from cdc.data.bar_dataset import BAR
        dataset = BAR(cfg['data']['val_path'], train=False, transform=transform)
        
    elif cfg['data']['dataset'] in ["svhn"]:
        from torchvision.datasets import SVHN
        dataset = SVHN(cfg['data']['val_path'], 'test', transform)
    
    elif cfg['data']['dataset'] in ["places365"]:
        from torchvision.datasets import Places365
        dataset = Places365(cfg['data']['val_path'], 'val', transform=transform)
        
    elif cfg['data']['dataset'] in ["SUN397"]:
        from torchvision.datasets import SUN397
        dataset = SUN397(cfg['data']['val_path'], transform=transform)
        
    else:
        raise ValueError('Invalid validation dataset {}'.format(cfg['data']['dataset']))
    
    return dataset

def get_train_dataloader(cfg, dataset, is_drop_last = True, is_shuffle = True):
    return torch.utils.data.DataLoader(dataset, num_workers=cfg['data']['num_workers'],
            batch_size=cfg['optimizer']['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=is_drop_last, shuffle=is_shuffle, persistent_workers=True)
    
def get_val_dataloader(cfg, dataset, batch=500):
    return torch.utils.data.DataLoader(dataset, num_workers=cfg['data']['num_workers'],
            batch_size=batch, pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)

def get_model(cfg, pretrain=None):
    from cdc.backbones.models import ClusteringModel, CCModel, TCLModel, DivClust_CCModel
    model = ClusteringModel(cfg)
    if pretrain:
        state = torch.load(cfg['pretext_model'], map_location='cpu')
        if cfg['method'] == 'cdcv2':
            #import pdb;pdb.set_trace()
            model.load_state_dict(state['state_dict'], strict=False)

        elif cfg['method'] == 'tcl':
            if cfg['data']['dataset'] == 'cmnist':
                model = TCLModel(cfg, cmnist=True)
            else:
                model = TCLModel(cfg)
                missing = model.load_state_dict(state['state_dict'], strict=False)
                assert (set(missing[0]) == {
                    "instance_projector.0.weight", "instance_projector.0.bias","instance_projector.0.running_mean","instance_projector.0.running_var",
                    "instance_projector.2.weight", "instance_projector.2.bias",
                    "instance_projector.3.weight","instance_projector.3.bias","instance_projector.3.running_mean","instance_projector.3.running_var",
                    "instance_projector.5.weight", "instance_projector.5.bias", 
                    "cluster_projector.0.weight", "cluster_projector.0.bias", "cluster_projector.0.running_mean", "cluster_projector.0.running_var",
                    "cluster_projector.2.weight", "cluster_projector.2.bias", 
                    "cluster_projector.3.weight", "cluster_projector.3.bias", "cluster_projector.3.running_mean", "cluster_projector.3.running_var",
                    "cluster_projector.5.weight", "cluster_projector.5.bias", 
                    })
        elif cfg['method'] == 'cc':
            model = CCModel(cfg)
            missing = model.load_state_dict(state['state_dict'], strict=False)
            assert (set(missing[0]) == {
                "instance_projector.0.weight", "instance_projector.0.bias",
                "instance_projector.2.weight", "instance_projector.2.bias",
                "cluster_projector.0.weight", "cluster_projector.0.bias",
                "cluster_projector.2.weight", "cluster_projector.2.bias"})
        elif cfg['method'] == 'divclust_cc':
            model = DivClust_CCModel(cfg)
            missing = model.load_state_dict(state['state_dict'], strict=False)
            assert (set(missing[0]) == {
                "instance_projector.0.weight", "instance_projector.0.bias",
                "instance_projector.2.weight", "instance_projector.2.bias",
                "cluster_projector.0.weight", "cluster_projector.0.bias",
                "cluster_projector.2.weight", "cluster_projector.2.bias"})
        elif cfg['method'] == 'scan':
            from cdc.backbones.models import SCANModel
            model = SCANModel(cfg)
            """ model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()}, strict=False) """
            try:
                missing = model.load_state_dict(state['state_dict'], strict=False)
            except:
                model.load_state_dict({k.replace('module.', ''): v for k, v in state['model'].items()}, strict=False)
        elif cfg['method'] == 'su':
            model.load_state_dict(state['state_dict'], strict=False)
            
        else:
            raise NotImplementedError
        
        
    elif pretrain and not os.path.exists(cfg['pretext_model']):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(cfg['pretext_model']))
    elif cfg['method'] == 'tcl':
        model = TCLModel(cfg)
    else:
        pass
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    return model

def get_optimizer(cfg, model1):
    """ params = filter(lambda p: p.requires_grad, model1.parameters())
    params = [
        {'params': model1.module.backbone.parameters(), 'lr': cfg['optimizer']['lr']},
        {'params': model1.module.cluster_head.parameters(), 'lr': cfg['optimizer']['lr']}
    ] """
    params = [
            {'params': filter(lambda p: p.requires_grad, model1.parameters()), 'lr': cfg['optimizer']['lr']},
        ]
    if cfg['optimizer']['name'] == 'sgd':
        optimizer = torch.optim.SGD(params, **cfg['optimizer']['kwargs'])
    elif cfg['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(params, **cfg['optimizer']['kwargs'])
    return optimizer

def get_criterion(cfg):
    if cfg['method'] in ['fixed','dual_fixed','dual_calibration']:
        # from losses.losses import FixedLoss
        # criterion = FixedLoss(**cfg['criterion_kwargs'])
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        return criterion
    elif cfg['method'] == 'calibration' or \
            cfg['method'] == 'shallow':
        # from losses.losses import ConfidenceBasedCE
        # criterion1 = ConfidenceBasedCE(cfg['confidence_threshold'], cfg['criterion_kwargs']['apply_class_balancing'])
        # criterion2 = ConfidenceBasedCE(cfg['confidence_threshold'], cfg['criterion_kwargs']['apply_class_balancing'])
        # return criterion1, criterion2
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        return criterion
    elif cfg['method'] == 'mcd':
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        return criterion
    elif cfg['criterion']['name'] == 'ce' \
            or cfg['criterion']['name'] == 'cc'\
            or cfg['criterion']['name'] == 'scan':
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        return criterion
    # elif cfg['method'] == 'scan':
    #
    # elif cfg['method'] == 'cc':

    else:
        pass
        # raise ValueError('Invalid criterion ')
        
        
        
def adjust_learning_rate(cfg, optimizer, epoch):
    lr = cfg['optimizer']['lr']
    if cfg['scheduler']['name'] == 'cosine':
        eta_min = lr * (cfg['scheduler']['kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / cfg['max_epochs'])) / 2
    elif cfg['scheduler']['name'] == 'step':
        steps = np.sum(epoch > np.array(cfg['scheduler']['kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (cfg['scheduler']['kwargs']['lr_decay_rate'] ** steps)
    elif cfg['scheduler']['name'] == 'linear_warmup':
        margin = (lr-cfg['scheduler']['kwargs']['lr_start'])/(cfg['scheduler']['kwargs']['warmup_steps'])
        lr = min(cfg['scheduler']['kwargs']['lr_start']+epoch*margin,lr)
    elif cfg['scheduler']['name'] == 'exp':
        ratio = cfg['optimizer']['lr'] / cfg['scheduler']['kwargs']['lr_start']
        margin = np.log10(ratio) / (cfg['scheduler']['kwargs']['warmup_steps'])
        lr = min(cfg['scheduler']['kwargs']['lr_start'] * 10**(epoch*margin), lr)
    elif cfg['scheduler']['name'] == 'constant_warmup':
        if epoch < cfg['scheduler']['kwargs']['warmup_steps']:
            lr = cfg['scheduler']['kwargs']['lr_start']
        # if epoch>20:
        #     lr =0.00001
    # elif cfg['scheduler']['name'] == 'warmup':
    #     warmup_steps = cfg['scheduler']['kwargs']['warmup_steps']
    #     step_length = cfg['scheduler']['kwargs']['step_length']
    #     lr = lr * min((step_length*epoch + 1) / (step_length*warmup_steps), (((step_length*warmup_steps) ** 0.5) / (rate*epoch + 1)))
    elif cfg['scheduler']['name'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule')

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # if len(optimizer.param_groups)==2:
    #     optimizer.param_groups[1]['lr'] = lr
    # else:
    #     optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups)==2:
        optimizer.param_groups[0]['lr'] = lr
    return lr
