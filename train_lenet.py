import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# =====================================================================
# 1. PREPROCESSING & DATA AUGMENTATION
# =====================================================================
# We resize to 64x64 (instead of classic LeNet's 32x32) to keep FEA details.
# We also apply random flips to help the model generalize with small data.
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load data from folders
data_dir = "./CNN_dataset"
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
print(f"Class mapping found: {full_dataset.class_to_idx}")
print(f"Total images loaded: {len(full_dataset)}")

# =====================================================================
# 2. DATASET SPLITTING (70% Train, 15% Val, 15% Test)
# =====================================================================
total_count = len(full_dataset)
train_count = int(0.7 * total_count)
val_count = int(0.15 * total_count)
test_count = total_count - train_count - val_count

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_count, val_count, test_count],
    generator=torch.Generator().manual_seed(42) # For reproducible splits
)

# Create Data Loaders (Batch size 4 because the dataset is small)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# =====================================================================
# 3. LENET ARCHITECTURE (Adapted for 3 Classes & 64x64 RGB Images)
# =====================================================================
class LeNet5(nn.Module):
    def __init__(self, num_classes=3):
        super(LeNet5, self).__init__()
        # Input: 3 channels (RGB) x 64 x 64
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: 6 x 30 x 30
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16 x 13 x 13
        )
        
        # Flattened features: 16 * 13 * 13 = 2704
        self.classifier = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes) # Outputs raw scores (logits) for 3 classes
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        logits = self.classifier(x)
        return logits

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================================================================
# 4. MODEL TRAINING LOOP
# =====================================================================
epochs = 20
print(f"\nStarting Training on device: {device}...")

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

# =====================================================================
# 5. MODEL TESTING
# =====================================================================
print("\n--- Running Final Evaluation on Test Split ---")
model.eval()
test_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels.data)

test_acc = test_correct.double() / len(test_loader.dataset)
print(f"Final Test Accuracy: {test_acc:.2%}")

# Save the trained weights
torch.save(model.state_dict(), "lenet_fea_anomaly.pth")
print("Model weights successfully saved to 'lenet_fea_anomaly.pth'!")