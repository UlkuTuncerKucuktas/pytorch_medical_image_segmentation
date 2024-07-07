import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ImageLabelDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None, label_transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform

        self.image_files = sorted([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
        self.label_files = sorted([f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))])

        self.image_files = [f for f in self.image_files if f in self.label_files]
        self.label_files = [f for f in self.label_files if f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_files[idx])
        label_name = os.path.join(self.label_folder, self.label_files[idx])

        image = Image.open(img_name).convert('RGB')
        label = Image.open(label_name).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.label_transform:
            label = self.label_transform(label)

        return image, label
