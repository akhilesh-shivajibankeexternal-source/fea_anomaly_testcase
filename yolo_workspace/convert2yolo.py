import json
import os
from PIL import Image

# ==========================================
# 1. SETUP PATHS AND MAPPINGS
# ==========================================
# Change these paths to point to your actual folders
IMAGE_DIR = "./raw_images"       # Folder containing your 240 images
JSON_DIR = "./pixlab_jsons"      # Folder containing your 240 JSON files
OUTPUT_DIR = "./yolo_labels"     # Where the new .txt files will be saved

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map your PixLab label names to YOLO integer IDs
# Using .lower() to avoid case-sensitivity issues (e.g., "Duplicate_element" vs "duplicate_element")
CLASS_MAPPING = {
    "duplicate_element": 0,
    "duplicate_elem_anomaly": 0,
    "in_plane_distortion": 1,
    "in_plane_anomaly": 1,
    "in_plane": 1,
    "out_plane_anomaly": 2,
    "out_of_plane_distortion": 2,
    "out_of_plane": 2,
    "out_of_plan_distortion": 2  # Added to catch the typo!
}

# ==========================================
# 2. CONVERSION LOGIC
# ==========================================
def convert_pixlab_to_yolo():
    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files to process.")
    
    success_count = 0

    for json_filename in json_files:
        # Construct full paths
        json_path = os.path.join(JSON_DIR, json_filename)
        base_name = os.path.splitext(json_filename)[0]
        
        # Try to find the corresponding image (assuming .png, change to .jpg if needed)
        image_path = os.path.join(IMAGE_DIR, f"{base_name}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(IMAGE_DIR, f"{base_name}.jpg") # Fallback to jpg
            
        if not os.path.exists(image_path):
            print(f"Warning: Could not find matching image for {json_filename}. Skipping.")
            continue
            
        # Get image dimensions for normalization
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        # Read the PixLab JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        yolo_lines = []
        
        # Parse each bounding box in the JSON
        for annotation in data:
            # 1. Get the class ID
            label_name = annotation.get("labels", {}).get("labelName", "").lower()
            if label_name not in CLASS_MAPPING:
                print(f"Warning: Unrecognized label '{label_name}' in {json_filename}.")
                continue
            class_id = CLASS_MAPPING[label_name]
            
            # 2. Get the absolute bounding box coordinates
            rect = annotation.get("rectMask", {})
            if not rect:
                continue
                
            x_min = rect["xMin"]
            y_min = rect["yMin"]
            box_width = rect["width"]
            box_height = rect["height"]
            
            # 3. Calculate YOLO format (center x, center y, width, height)
            x_center = x_min + (box_width / 2.0)
            y_center = y_min + (box_height / 2.0)
            
            # 4. Normalize by image dimensions
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = box_width / img_width
            height_norm = box_height / img_height
            
            # Ensure values are strictly between 0 and 1
            x_center_norm = max(0.0, min(1.0, x_center_norm))
            y_center_norm = max(0.0, min(1.0, y_center_norm))
            width_norm = max(0.0, min(1.0, width_norm))
            height_norm = max(0.0, min(1.0, height_norm))
            
            # Format to 6 decimal places
            yolo_line = f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
            yolo_lines.append(yolo_line)
            
        # Write to the output .txt file
        output_txt_path = os.path.join(OUTPUT_DIR, f"{base_name}.txt")
        with open(output_txt_path, 'w') as out_file:
            out_file.write("\n".join(yolo_lines))
            
        success_count += 1

    print(f"\nSuccessfully converted {success_count} files to YOLO format!")
    print(f"Your YOLO labels are waiting in: {OUTPUT_DIR}")

if __name__ == "__main__":
    convert_pixlab_to_yolo()