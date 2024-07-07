from dataset.image_label_dataset import ImageLabelDataset
from models.unet import Unet
import torch
from torch.utils.data import random_split, DataLoader

image_folder = "/content/dataset/content/latest_dataset/images"
label_folder = "/content/dataset/content/latest_dataset/labels"
dataset = ImageLabelDataset(image_folder,label_folder)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size


train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)