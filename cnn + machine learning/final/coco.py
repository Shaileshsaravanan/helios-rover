import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from pycocotools.coco import COCO
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, Rotate, 
    RandomBrightnessContrast, GaussianBlur, GaussNoise
)

# Define preprocessing and augmentation transformations
def preprocess_image(image_path, target_size=(640, 640)):
    print(f"Loading image from {image_path}")
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    print(f"Image resized to {target_size}")
    return img

def transform_fn(image):
    print("Applying transformations")
    transform = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        Rotate(limit=15, p=0.5),
        RandomBrightnessContrast(p=0.2),
        GaussianBlur(blur_limit=(1, 3), p=0.2),  # Updated GaussianBlur configuration
        GaussNoise(var_limit=(0, 0.05*255**2), p=0.2)  # Replace IAAAdditiveGaussianNoise
    ])
    augmented = transform(image=image)
    print("Transformations applied")
    return augmented['image']

# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, coco, img_dir, transform=None):
        print("Initializing dataset")
        self.coco = coco
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = coco.getImgIds()
        print(f"Dataset contains {len(self.img_ids)} images")
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = preprocess_image(img_path)
        if self.transform:
            img = self.transform(img)
        img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W) format
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        return img

# Model definition
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        print(f"Initializing model with {num_classes} classes")
        super(CustomModel, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# Load COCO annotations
print("Loading COCO annotations")
train_coco = COCO('/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/dataset/train/_annotations.coco.json')
valid_coco = COCO('/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/dataset/valid/_annotations.coco.json')
test_coco = COCO('/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/dataset/test/_annotations.coco.json')
print("COCO annotations loaded")

# Create datasets and dataloaders
print("Creating datasets and dataloaders")
train_dataset = CustomDataset(coco=train_coco, img_dir='/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/dataset/train', transform=transform_fn)
valid_dataset = CustomDataset(coco=valid_coco, img_dir='/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/dataset/valid', transform=transform_fn)
test_dataset = CustomDataset(coco=test_coco, img_dir='/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/dataset/test', transform=transform_fn)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("Datasets and dataloaders created")

# Initialize model, loss function, and optimizer
print("Initializing model, loss function, and optimizer")
num_classes = 9  # Example number of classes
model = CustomModel(num_classes=num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(f"Model moved to device: {device}")

# Training loop
num_epochs = 20
print(f"Starting training for {num_epochs} epochs")
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for images in train_loader:
        images = images.to(device)
        # Dummy targets for example; replace with actual targets
        targets = torch.randint(0, num_classes, (images.size(0),), device=device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    # Validation
    model.eval()
    with torch.no_grad():
        valid_loss = 0.0
        for images in valid_loader:
            images = images.to(device)
            # Dummy targets for example; replace with actual targets
            targets = torch.randint(0, num_classes, (images.size(0),), device=device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            valid_loss += loss.item() * images.size(0)
        
        epoch_valid_loss = valid_loss / len(valid_loader.dataset)
        print(f'Validation Loss: {epoch_valid_loss:.4f}')

# Testing (optional, example only)
print("Saving the model")
model.eval()
model_path = '/Users/shaileshsaravanan/Documents/VS Code/helios-rover/cnn + machine learning/final/custom_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': epoch_loss,
}, model_path)
print(f'Model saved to {model_path}')

print("Evaluating model on test set")
with torch.no_grad():
    for images in test_loader:
        images = images.to(device)
        outputs = model(images)
        # Process outputs as needed for evaluation
print("Test evaluation complete")
