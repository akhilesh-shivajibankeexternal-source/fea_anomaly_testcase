import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# =====================================================================
# 1. RECREATE THE ARCHITECTURE & LOAD WEIGHTS
# =====================================================================
class LeNet5(nn.Module):
    def __init__(self, num_classes=3):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5(num_classes=3).to(device)
model.load_state_dict(torch.load("lenet_fea_anomaly.pth", map_location=device))
model.eval()

# =====================================================================
# 2. RECREATE DATA SPLITS (Must match training exactly)
# =====================================================================
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # Note: Removed random flips for evaluation so we test on clean images
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = "./CNN_dataset"
full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
class_names = {v: k for k, v in full_dataset.class_to_idx.items()} # Reverse mapping

total_count = len(full_dataset)
train_count = int(0.7 * total_count)
val_count = int(0.15 * total_count)
test_count = total_count - train_count - val_count

# CRITICAL: Use the exact same seed (42) to ensure the splits are identical
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_count, val_count, test_count],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=total_count, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=total_count, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=total_count, shuffle=False)

# =====================================================================
# 3. HELPER FUNCTIONS FOR VISUALIZATION
# =====================================================================
def get_predictions(loader):
    """Extracts all true labels and model predictions from a dataloader."""
    all_preds = []
    all_labels = []
    images_list = []
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
    """Plots a single confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[class_names[i] for i in range(3)],
                yticklabels=[class_names[i] for i in range(3)])
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

def unnormalize(img):
    """Reverts the normalization so images look normal when plotted."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = img.numpy() * std + mean
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0)) # Change (C, H, W) to (H, W, C) for Matplotlib

# =====================================================================
# 4. EXECUTE AND PLOT
# =====================================================================
print("Extracting predictions...")
train_true, train_pred, _ = get_predictions(train_loader)
val_true, val_pred, _ = get_predictions(val_loader)
test_true, test_pred, test_images = get_predictions(test_loader)

# --- Plot 1: Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_confusion_matrix(train_true, train_pred, "Training Confusion Matrix", axes[0])
plot_confusion_matrix(val_true, val_pred, "Validation Confusion Matrix", axes[1])
plot_confusion_matrix(test_true, test_pred, "Test Confusion Matrix", axes[2])
plt.tight_layout()
plt.show()

# # --- Plot 2: Visual Results on Test Set ---
# print("Plotting visual results from the Test Set...")
# num_images_to_show = min(8, len(test_images)) # Show up to 8 images
# fig_img, axes_img = plt.subplots(2, 4, figsize=(15, 8))
# axes_img = axes_img.flatten()

# for i in range(num_images_to_show):
#     img = test_images[i]
#     true_label = class_names[test_true[i]]
#     pred_label = class_names[test_pred[i]]
    
#     # Determine color (Green for correct, Red for wrong)
#     color = "green" if true_label == pred_label else "red"
    
#     axes_img[i].imshow(unnormalize(img))
#     axes_img[i].set_title(f"Act: {true_label}\nPred: {pred_label}", color=color, fontsize=10)
#     axes_img[i].axis('off')

# # Hide any unused subplots if test set is smaller than 8
# for j in range(num_images_to_show, 8):
#     axes_img[j].axis('off')

# plt.tight_layout()
# plt.show()


# --- Plot 2: Visual Results on Test Set ---
print("Plotting visual results from the Test Set...")

# 1. Extract the original file names for the test dataset
test_indices = test_dataset.indices
test_filenames = [os.path.basename(full_dataset.samples[i][0]) for i in test_indices]

num_images_to_show = min(8, len(test_images)) # Show up to 8 images
fig_img, axes_img = plt.subplots(2, 4, figsize=(15, 8))
axes_img = axes_img.flatten()

for i in range(num_images_to_show):
    img = test_images[i]
    true_label = class_names[test_true[i]]
    pred_label = class_names[test_pred[i]]
    file_name = test_filenames[i] # Grab the filename we extracted
    
    # Determine color (Green for correct, Red for wrong)
    color = "green" if true_label == pred_label else "red"
    
    axes_img[i].imshow(unnormalize(img))
    
    # Updated title to show the file name
    axes_img[i].set_title(f"Act: {file_name}\nPred: {pred_label}", color=color, fontsize=10)
    axes_img[i].axis('off')

# Hide any unused subplots if test set is smaller than 8
for j in range(num_images_to_show, 8):
    axes_img[j].axis('off')

plt.tight_layout()
plt.show()