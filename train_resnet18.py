import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# =====================================================================
# 1. PREPROCESSING (Upgraded to 224x224 for ResNet)
# =====================================================================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Crucial for capturing thin mesh lines
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=15), # Add slight rotations
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Add slight lighting changes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

data_dir = "./CNN_dataset"
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
print(f"Class mapping found: {full_dataset.class_to_idx}")

# =====================================================================
# 2. DATASET SPLITTING (Seed 42 to match previous splits exactly)
# =====================================================================
# total_count = len(full_dataset)
# train_count = int(0.7 * total_count)
# val_count = int(0.15 * total_count)
# test_count = total_count - train_count - val_count

# train_dataset, val_dataset, test_dataset = random_split(
#     full_dataset, [train_count, val_count, test_count],
#     generator=torch.Generator().manual_seed(42)
# )

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# =====================================================================
# 2. STRATIFIED DATASET SPLITTING (75% Train, 15% Val, 10% Test)
# =====================================================================
# Extract all labels from the dataset to use for stratification
targets = full_dataset.targets
indices = list(range(len(full_dataset)))

# SPLIT 1: Separate Training (75%) from the rest (25%)
train_idx, temp_idx, _, temp_targets = train_test_split(
    indices, 
    targets, 
    test_size=0.25, 
    stratify=targets, 
    random_state=42
)

# SPLIT 2: Split the remaining 25% into Validation (15%) and Test (10%)
# 10% is 40% of the 25% temp set (0.4 * 0.25 = 0.10)
val_idx, test_idx = train_test_split(
    temp_idx, 
    test_size=0.40, 
    stratify=temp_targets, 
    random_state=42
)

# Create PyTorch Subsets using the stratified indices
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)
test_dataset = Subset(full_dataset, test_idx)

# Create Data Loaders (Keep batch size small)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# =====================================================================
# 3. RESNET-18 TRANSFER LEARNING ARCHITECTURE
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet18
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Replace the final fully connected layer (fc) for our 3 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3) 

model = model.to(device)
criterion = nn.CrossEntropyLoss()
# Use a smaller learning rate for fine-tuning pre-trained models
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4) 

# =====================================================================
# 4. MODEL TRAINING LOOP
# =====================================================================
epochs = 25
print(f"\nStarting ResNet-18 Training on device: {device}...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    
    # Validation phase
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
            
    val_acc = val_correct.double() / len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc:.2%}")

# Save the trained weights
torch.save(model.state_dict(), "resnet18_fea_anomaly.pth")
print("Model weights successfully saved to 'resnet18_fea_anomaly.pth'!")