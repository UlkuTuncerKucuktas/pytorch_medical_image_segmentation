import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class ImageLabelDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform

        self.image_files = sorted([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
        self.label_files = sorted([f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))])

        self.image_files = [f for f in self.image_files if f in self.label_files]
        self.label_files = [f for f in self.label_files if f in self.image_files]

        if self.transform is None:
            self.transform = A.Compose([
                A.Resize(320,640),
                A.ToFloat(max_value=255.0),  # Ensure images are in [0, 1] range
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        label_name = os.path.join(self.label_folder, self.label_files[idx])

        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')

        image = np.array(image)
        label = np.array(label) / 255

        if self.transform:
            augmented = self.transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask']

        return image, label