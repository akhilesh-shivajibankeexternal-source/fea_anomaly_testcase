import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# =====================================================================
# 1. LOAD RESNET ARCHITECTURE & WEIGHTS
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize base ResNet18 and modify the final layer to match training
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

# Load the saved weights
model.load_state_dict(torch.load("resnet18_fea_anomaly.pth", map_location=device))
model = model.to(device)
model.eval()

# # =====================================================================
# # 2. RECREATE DATA SPLITS (Must match training exactly)
# # =====================================================================
# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)), # Match ResNet resolution
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# data_dir = "./CNN_dataset"
# full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
# class_names = {v: k for k, v in full_dataset.class_to_idx.items()}
# target_names_list = [class_names[i] for i in range(3)]

# total_count = len(full_dataset)
# train_count = int(0.75 * total_count)
# val_count = int(0.15 * total_count)
# test_count = total_count - train_count - val_count

# # Same seed ensures identical test images
# train_dataset, val_dataset, test_dataset = random_split(
#     full_dataset, [train_count, val_count, test_count],
#     generator=torch.Generator().manual_seed(42)
# )

# train_loader = DataLoader(train_dataset, batch_size=total_count, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=total_count, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=total_count, shuffle=False)


# =====================================================================
# 2. RECREATE DATA SPLITS (Must match training exactly)
# =====================================================================
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Match ResNet resolution
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = "./CNN_dataset"
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
class_names = {v: k for k, v in full_dataset.class_to_idx.items()}
target_names_list = [class_names[i] for i in range(3)]

total_count = len(full_dataset)

# Extract all labels from the dataset to use for stratification
targets = full_dataset.targets
indices = list(range(total_count))

# SPLIT 1: Separate Training (75%) from the rest (25%)
train_idx, temp_idx, _, temp_targets = train_test_split(
    indices, 
    targets, 
    test_size=0.25, 
    stratify=targets, 
    random_state=42
)

# SPLIT 2: Split the remaining 25% into Validation (15%) and Test (10%)
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

# Create Data Loaders (Keep batch size as total_count for evaluation)
train_loader = DataLoader(train_dataset, batch_size=total_count, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=total_count, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=total_count, shuffle=False)


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================
def get_predictions(loader):
    all_preds, all_labels, images_list = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            images_list.extend(images.cpu())
    return all_labels, all_preds, images_list

def plot_confusion_matrix(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=target_names_list,
                yticklabels=target_names_list)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

def unnormalize(img):
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img.numpy() * std + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))

# =====================================================================
# 4. EXECUTE, PRINT REPORT, AND PLOT
# =====================================================================
print("Extracting predictions...")
train_true, train_pred, _ = get_predictions(train_loader)
val_true, val_pred, _ = get_predictions(val_loader)
test_true, test_pred, test_images = get_predictions(test_loader)

# --- Print Classification Report ---
print("\n" + "="*50)
print("CLASSIFICATION REPORT (TEST SET)")
print("="*50)
print(classification_report(test_true, test_pred, target_names=target_names_list))
print("="*50 + "\n")

# --- Plot 1: Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_confusion_matrix(train_true, train_pred, "Training Confusion Matrix", axes[0])
plot_confusion_matrix(val_true, val_pred, "Validation Confusion Matrix", axes[1])
plot_confusion_matrix(test_true, test_pred, "Test Confusion Matrix", axes[2])
plt.tight_layout()
plt.show()

# --- Plot 2: Visual Results on Test Set (With Filenames) ---
print("Plotting visual results from the Test Set...")

test_indices = test_dataset.indices
test_filenames = [os.path.basename(full_dataset.samples[i][0]) for i in test_indices]

num_images_to_show = min(8, len(test_images))
fig_img, axes_img = plt.subplots(2, 4, figsize=(15, 8))
axes_img = axes_img.flatten()

for i in range(num_images_to_show):
    img = test_images[i]
    true_label = class_names[test_true[i]]
    pred_label = class_names[test_pred[i]]
    file_name = test_filenames[i]
    
    color = "green" if true_label == pred_label else "red"
    
    axes_img[i].imshow(unnormalize(img))
    axes_img[i].set_title(f"Act: {file_name}\nPred: {pred_label}", color=color, fontsize=10)
    axes_img[i].axis('off')

for j in range(num_images_to_show, 8):
    axes_img[j].axis('off')

plt.tight_layout()
plt.show()