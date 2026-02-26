import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CIFAR10CDataset(Dataset):
    def __init__(self, root_dir, corruption_type, severity=None, transform=None):
        """
        Args:
            root_dir (string): Directory with the CIFAR-10-C dataset.
            corruption_type (string): Type of corruption (e.g., 'gaussian_noise', 'fog', etc.).
            severity (int, optional): Severity level (1-5). If None, uses all severities.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.root_dir = root_dir
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        
        # Load the corrupted images
        data_path = os.path.join(root_dir, f"{corruption_type}.npy")
        labels_path = os.path.join(root_dir, "labels.npy")
        
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(f"Could not find {data_path} or {labels_path}")
            
        # Load data and labels
        self.data = np.load(data_path)       # Shape: (50000, 32, 32, 3)
        self.labels = np.load(labels_path)   # Shape: (50000,)
        
        # CIFAR-10-C has 10,000 test images per severity level (1-5)
        # Each severity level has images at indices: severity_level*10000:(severity_level+1)*10000
        if severity is not None:
            if not 1 <= severity <= 5:
                raise ValueError("Severity level must be between 1 and 5")
            
            # Extract data for the specified severity level (0-indexed in the array)
            start_idx = (severity - 1) * 10000
            end_idx = severity * 10000
            self.data = self.data[start_idx:end_idx]
            self.labels = self.labels[start_idx:end_idx]
            self.targets = self.labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32) / 255.0
        
        return image, label


def get_cifar10c_dataloader(root_dir, corruption_type, severity=None, batch_size=128, num_workers=4, shuffle=False):
    """
    Create a dataloader for a specific CIFAR-10-C corruption type and severity.
    
    Args:
        root_dir (string): Path to CIFAR-10-C dataset.
        corruption_type (string): Type of corruption.
        severity (int, optional): Severity level (1-5). If None, uses all severities.
        batch_size (int): Batch size for dataloader.
        num_workers (int): Number of workers for dataloader.
        shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
        DataLoader: PyTorch DataLoader for the specified CIFAR-10-C subset.
    """
    # Standard CIFAR-10 normalization
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    dataset = CIFAR10CDataset(
        root_dir=root_dir,
        corruption_type=corruption_type,
        severity=severity,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_all_cifar10c_dataloaders(root_dir, severity=None, batch_size=128):
    """
    Create dataloaders for all corruption types in CIFAR-10-C.
    
    Args:
        root_dir (string): Path to CIFAR-10-C dataset.
        severity (int, optional): Severity level (1-5). If None, uses all severities.
        batch_size (int): Batch size for dataloaders.
    
    Returns:
        dict: Dictionary mapping corruption types to their respective DataLoaders.
    """
    # Standard corruption types in CIFAR-10-C
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    dataloaders = {}
    for corruption in corruptions:
        # Check if the corruption file exists
        if os.path.exists(os.path.join(root_dir, f"{corruption}.npy")):
            dataloaders[corruption] = get_cifar10c_dataloader(
                root_dir, corruption, severity, batch_size
            )
        else:
            print(f"Warning: Corruption file '{corruption}.npy' not found.")
    
    return dataloaders



class CIFAR20CDataset(Dataset):
    def __init__(self, root_dir, corruption_type, severity=None, transform=None):
        """
        Args:
            root_dir (string): Directory with the CIFAR-20-C dataset.
            corruption_type (string): Type of corruption (e.g., 'gaussian_noise', 'fog', etc.).
            severity (int, optional): Severity level (1-5). If None, uses all severities.
            transform (callable, optional): Optional transform to be applied on images.
        """
        self.root_dir = root_dir
        self.corruption_type = corruption_type
        self.severity = severity
        self.transform = transform
        
        # Define CIFAR-100 superclass mapping (20 superclasses)
        self.superclass_mapping = {
            'aquatic_mammals': [4, 30, 55, 72, 95],
            'fish': [1, 32, 67, 73, 91],
            'flowers': [54, 62, 70, 82, 92],
            'food_containers': [9, 10, 16, 28, 61],
            'fruit_and_vegetables': [0, 51, 53, 57, 83],
            'household_electrical_devices': [22, 39, 40, 86, 87],
            'household_furniture': [5, 20, 25, 84, 94],
            'insects': [6, 7, 14, 18, 24],
            'large_carnivores': [3, 42, 43, 88, 97],
            'large_man-made_outdoor_things': [12, 17, 37, 68, 76],
            'large_natural_outdoor_scenes': [23, 33, 49, 60, 71],
            'large_omnivores_and_herbivores': [15, 19, 21, 31, 38],
            'medium_mammals': [34, 63, 64, 66, 75],
            'non-insect_invertebrates': [26, 45, 77, 79, 99],
            'people': [2, 11, 35, 46, 98],
            'reptiles': [27, 29, 44, 78, 93],
            'small_mammals': [36, 50, 65, 74, 80],
            'trees': [47, 52, 56, 59, 96],
            'vehicles_1': [8, 13, 48, 58, 90],
            'vehicles_2': [41, 69, 81, 85, 89]
        }
        
        # Create reverse mapping from CIFAR-100 class to superclass
        self.class_to_superclass = {}
        for superclass_idx, (superclass, classes) in enumerate(self.superclass_mapping.items()):
            for class_idx in classes:
                self.class_to_superclass[class_idx] = superclass_idx
        
        # Load the corrupted images
        data_path = os.path.join(root_dir, f"{corruption_type}.npy")
        labels_path = os.path.join(root_dir, "labels.npy")
        
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(f"Could not find {data_path} or {labels_path}")
            
        # Load data and labels
        self.data = np.load(data_path)       # Shape: (50000, 32, 32, 3)
        self.fine_labels = np.load(labels_path)   # Shape: (50000,)
        
        # Convert fine labels (0-99) to superclass labels (0-19)
        self.superclass_labels = np.array([self.class_to_superclass[label] for label in self.fine_labels])
        
        # CIFAR-100-C has 10,000 test images per severity level (1-5)
        if severity is not None:
            if not 1 <= severity <= 5:
                raise ValueError("Severity level must be between 1 and 5")
            
            # Extract data for the specified severity level (0-indexed in the array)
            start_idx = (severity - 1) * 10000
            end_idx = severity * 10000
            self.data = self.data[start_idx:end_idx]
            self.superclass_labels = self.superclass_labels[start_idx:end_idx]
    
    def __len__(self):
        return len(self.superclass_labels)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.superclass_labels[idx]
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32) / 255.0
        
        return image, label


def get_cifar20c_dataloader(root_dir, corruption_type, severity=None, batch_size=128, num_workers=4, shuffle=False):
    """
    Create a dataloader for a specific CIFAR-20-C corruption type and severity.
    
    Args:
        root_dir (string): Path to CIFAR-100-C dataset.
        corruption_type (string): Type of corruption.
        severity (int, optional): Severity level (1-5). If None, uses all severities.
        batch_size (int): Batch size for dataloader.
        num_workers (int): Number of workers for dataloader.
        shuffle (bool): Whether to shuffle the dataset.
    
    Returns:
        DataLoader: PyTorch DataLoader for the specified CIFAR-20-C subset.
    """
    # Standard CIFAR-100 normalization
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    dataset = CIFAR20CDataset(
        root_dir=root_dir,
        corruption_type=corruption_type,
        severity=severity,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def get_all_cifar20c_dataloaders(root_dir, severity=None, batch_size=128):
    """
    Create dataloaders for all corruption types in CIFAR-20-C.
    
    Args:
        root_dir (string): Path to CIFAR-100-C dataset.
        severity (int, optional): Severity level (1-5). If None, uses all severities.
        batch_size (int): Batch size for dataloaders.
    
    Returns:
        dict: Dictionary mapping corruption types to their respective DataLoaders.
    """
    # Standard corruption types in CIFAR-100-C (same as CIFAR-10-C)
    corruptions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
    ]
    
    dataloaders = {}
    for corruption in corruptions:
        # Check if the corruption file exists
        if os.path.exists(os.path.join(root_dir, f"{corruption}.npy")):
            dataloaders[corruption] = get_cifar20c_dataloader(
                root_dir, corruption, severity, batch_size
            )
        else:
            print(f"Warning: Corruption file '{corruption}.npy' not found.")
    
    return dataloaders


