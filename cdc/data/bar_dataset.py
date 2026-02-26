import os
from PIL import Image
from torch.utils.data import Dataset

class BAR(Dataset):
    """
    Biased Activity Recognition (BAR) Dataset without CSV labels.
    Images filenames must encode labels as: <action>_<unique_id>.<ext>
    Example: climbing_0.jpg

    Each action has a one-to-one corresponding background:
      climbing   ↔ rockwall
      diving     ↔ underwater
      fishing    ↔ watersurface
      racing     ↔ apavedtrack
      throwing   ↔ playingfield
      vaulting   ↔ sky

    Directory structure:
    BAR/
        train/
            climbing_0.jpg  # all training images
            ...
        test/
            climbing_326.jpg  # all testing images (bias-conflicting)
            ...
    """
    # lowercase action and background names
    action_classes = ['climbing', 'diving', 'fishing', 'racing', 'throwing', 'pole vaulting']
    background_classes = ['rockwall', 'underwater', 'watersurface', 'apavedtrack', 'playingfield', 'sky']

    def __init__(self, root, train=True, transform=None, target_transform=None, verbose=False):
        super().__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.verbose = verbose

        split = 'train' if train else 'test'
        # images are directly under train/ or test/
        self.images_dir = os.path.join(root, split)
        if not os.path.isdir(self.images_dir):
            raise RuntimeError(f"Directory not found: {self.images_dir}")

        self.samples = []  # list of (image_path, action_idx, background_idx)
        for fname in sorted(os.listdir(self.images_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                continue
            name = os.path.splitext(fname)[0]
            parts = name.split('_', 1)  # split into action and id
            if len(parts) != 2:
                if self.verbose:
                    print(f"Skipping '{fname}': filename must be '<action>_<id>.<ext>'")
                continue
            action = parts[0].lower()
            if action not in self.action_classes:
                if self.verbose:
                    print(f"Skipping '{fname}': unknown action '{action}'")
                continue
            action_idx = self.action_classes.index(action)
            # one-to-one mapping: background index equals action index
            background_idx = action_idx
            img_path = os.path.join(self.images_dir, fname)
            self.samples.append((img_path, action_idx, background_idx))

        if self.verbose:
            print(f"Loaded {len(self.samples)} samples from {self.images_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, action_idx, background_idx = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            action_idx = self.target_transform(action_idx)

        return image, action_idx, idx

# Example usage:
# from torchvision import transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
# train_dataset = BAR(root='/path/to/BAR', train=True, transform=transform, verbose=True)
# test_dataset  = BAR(root='/path/to/BAR', train=False, transform=transform, verbose=True)