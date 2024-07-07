import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from dataset.image_label_dataset import ImageLabelDataset
from models.unet import Unet
from losses.dice import DiceLoss
from metrics.dice import DiceScoreMetric

# Hyperparameters
num_epochs = 25
learning_rate = 0.001
batch_size = 16

# Dataset and DataLoader
image_folder = "/content/dataset/content/latest_dataset/images"
label_folder = "/content/dataset/content/latest_dataset/labels"
dataset = ImageLabelDataset(image_folder, label_folder)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
device = torch.device("cuda" )
model = Unet(in_channels=1, classes=1).to(device)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
dice_metric = DiceScoreMetric()

# Training loop
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {loss.item():.4f}')
    
    return running_loss / len(train_loader)

# Evaluation loop
def evaluate(model, test_loader, dice_metric):
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            score = dice_metric(outputs, labels)
            dice_scores.append(score)
    
    mean_dice = sum(dice_scores) / len(dice_scores)
    return mean_dice

# Main training and evaluation
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}')
    
    dice_score = evaluate(model, test_loader, dice_metric)
    print(f'Epoch [{epoch+1}/{num_epochs}], Validation Dice Score: {dice_score:.4f}')
