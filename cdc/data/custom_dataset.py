"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            try:
                self.val_transform = transform['val']
            except:
                pass
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        image, label = data[0], data[1]

        sample = {}
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)
        try:
            sample['val'] = self.val_transform(image)
        except:
            pass
        sample['index']= index
        sample['target']= label

        return sample
    
    
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        transform = dataset.transform

        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform

        dataset.transform = None
        self.dataset = dataset
        self.indices = indices  # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors + 1]
        assert (self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)

        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor_img, anchor_target = anchor[0], anchor[1]
        anchor_img = self.anchor_transform(anchor_img)
        neighbor_img, neighbor_target = neighbor[0], neighbor[1]
        neighbor_img = self.neighbor_transform(neighbor_img)

        output['image'] = anchor_img
        output['neighbor'] = neighbor_img
        output['index'] = index
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['target'] = anchor_target

        return output
    
    