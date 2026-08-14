"""
dataset.py
Preprocessing pipeline and PyTorch Dataset class for RetinaScan.

Handles:
- Resizing retinal images to 224x224 (ResNet50 input size)
- Normalizing using ImageNet mean/std (required for transfer learning)
- Loading images + labels via a custom PyTorch Dataset
"""

import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# ImageNet stats — required because we use ImageNet-pretrained ResNet50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard preprocessing pipeline: resize -> tensor -> normalize
preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])


class RetinaDataset(Dataset):
    """
    Custom PyTorch Dataset for loading retinal fundus images and their
    binary DR / No DR labels.

    Args:
        df: DataFrame with 'filename' and 'label' columns
            (filename should NOT include extension — .png is appended here)
        image_dir: folder path containing the actual image files
        transform: preprocessing pipeline to apply (default: preprocess_transform)
    """

    def __init__(self, df, image_dir, transform=preprocess_transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # NOTE: filenames in the CSV do not include the .png extension —
        # this was a real bug encountered during development (silent
        # cv2.imread failure), fixed by appending the extension here.
        img_path = os.path.join(self.image_dir, row['filename'] + ".png")

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image at: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(row['label'], dtype=torch.float32)
        return img, label