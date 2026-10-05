import os
import shutil
from sklearn.model_selection import train_test_split

# Setup paths (Adjust these if your folders are named differently)
IMAGE_DIR = "./raw_images"
LABEL_DIR = "./yolo_labels"
BASE_OUT_DIR = "./"

# Define YOLO directories
dirs = [
    "images/train", "images/val",
    "labels/train", "labels/val"
]

# Create the directories
for d in dirs:
    os.makedirs(os.path.join(BASE_OUT_DIR, d), exist_ok=True)

# Get all images (ignoring non-image files)
images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Split 80% Train, 20% Validation
train_imgs, val_imgs = train_test_split(images, test_size=0.20, random_state=42)

def move_files(file_list, split_name):
    for img_name in file_list:
        base_name = os.path.splitext(img_name)[0]
        txt_name = f"{base_name}.txt"
        
        # Source paths
        src_img = os.path.join(IMAGE_DIR, img_name)
        src_txt = os.path.join(LABEL_DIR, txt_name)
        
        # Destination paths
        dst_img = os.path.join(BASE_OUT_DIR, f"images/{split_name}", img_name)
        dst_txt = os.path.join(BASE_OUT_DIR, f"labels/{split_name}", txt_name)
        
        # Copy files if the label exists
        if os.path.exists(src_txt):
            shutil.copy(src_img, dst_img)
            shutil.copy(src_txt, dst_txt)

print("Moving Training files...")
move_files(train_imgs, "train")

print("Moving Validation files...")
move_files(val_imgs, "val")

print("Dataset successfully split and organized for YOLO!")