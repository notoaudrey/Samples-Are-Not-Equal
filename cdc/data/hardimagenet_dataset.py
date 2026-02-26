import matplotlib.image as mpimg
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import pickle
import os
from PIL import Image
import numpy as np

_IMAGENET_ROOT = '/nas/datasets/imagenet/'
_MASK_ROOT = '/nas/datasets/hardImageNet/'

with open(_MASK_ROOT+'meta/idx_to_wnid.pkl', 'rb') as f:
    idx_to_wnid = pickle.load(f)
wnid_to_idx = dict({v:k for k,v in idx_to_wnid.items()})
with open(_MASK_ROOT+'meta/hard_imagenet_idx.pkl', 'rb') as f:
    inet_idx = pickle.load(f)

def to_tens(img, mask):
    img, mask = [transforms.ToTensor()(x) for x in [img, mask]]
    return img, mask

class HardImageNet(Dataset):
    def __init__(self, split='val', aug=None, ft=False, balanced_subset=False):
        '''
        Returns original ImageNet index when ft is False, otherwise returns label between 0 and 14
        '''
        self.aug = aug if aug is not None else to_tens
        self.split = split
        self.balanced_subset = balanced_subset
        self.collect_mask_paths()
        self.num_classes = 15
        self.ft = ft

    def map_wnid_to_label(self, wnid):
        ind = wnid_to_idx[wnid]
        if self.ft:
            ind = inet_idx.index(ind)
        return ind

    def collect_mask_paths(self):
        if self.balanced_subset and self.split == 'train':
            # hard coded for now
            self.subset_size = 100

            with open(_MASK_ROOT+'paths_by_rank2.pkl', 'rb') as f:
                ranked_paths = pickle.load(f)
            paths = []
            for c in ranked_paths:
                cls_paths = ranked_paths[c]
                paths += cls_paths[:self.subset_size] + cls_paths[(-1*self.subset_size):]
            self.mask_paths = [_MASK_ROOT+'train/'+'_'.join(p.split('/')[-2:]) for p in paths]
            for p in self.mask_paths:
                if not os.path.exists(p):
                    self.mask_paths.remove(p)
        else:
            self.mask_paths = glob.glob(_MASK_ROOT + self.split+'/*')

    def __getitem__(self, ind):
        mask_path = self.mask_paths[ind]
        mask_path_suffix = mask_path.split('/')[-1]
        wnid = mask_path_suffix.split('_')[0]
        fname = mask_path_suffix[len(wnid)+1:] #if self.split == 'val' else mask_path_suffix

        img_path = os.path.join(_IMAGENET_ROOT, self.split, wnid, fname)
        img, mask = [Image.open(p) for p in [img_path, mask_path]]

        img, mask = self.aug(img, mask)

        if img.shape[0] > 3: #weird bug
            img, mask = [x[:3] for x in [img, mask]]

        class_ind = self.map_wnid_to_label(wnid)
        mask[mask > 0] = 1
        return img, mask, class_ind

    def __len__(self):
        return len(self.mask_paths)