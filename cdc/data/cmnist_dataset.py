import os
import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ColorMNISTDataset(Dataset):
    def __init__(self, h5_file_path, transform=None):
        """
        Args:
            h5_file_path (str): HDF5 文件路径（例如 'training.h5' 或 'testing.h5'）
            transform (callable, optional): 图像变换函数
        """
        self.h5_file_path = h5_file_path
        self.transform = transform
        self.length = self._get_length()

    def _get_length(self):
        with h5py.File(self.h5_file_path, 'r') as f:
            return len(f['images'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file_path, 'r') as f:
            image = f['images'][idx]
            label = f['labels'][idx]

        # 将图像转换为 PIL Image 或 Tensor，并应用变换
        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换为 Tensor
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, label, idx
