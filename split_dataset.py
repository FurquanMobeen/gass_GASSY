import os
import shutil
import random
from pathlib import Path
import glob
import re

def get_image_number(filename):
    """Extract number from image filename (frame_XXXX.jpg)"""
    match = re.search(r'frame_(\d+)\.jpg', filename)
    return int(match.group(1)) if match else None

def get_label_number(filename):
    """Extract number from label filename (ImageXXX.txt)"""
    match = re.search(r'Image(\d+)\.txt', filename)
    return int(match.group(1)) if match else None

def create_directories(base_path):
    """Create train, val, test directories with images and labels subdirectories"""
    for split in ['train', 'val', 'test']:
        for data_type in ['images', 'labels']:
            for camera in ['back_left', 'back_right', 'front', 'top']:
                dir_path = base_path / split / data_type / camera
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {dir_path}")

def match_files(images_dir, labels_dir):
    """Match image files with their corresponding label files"""
    matched_pairs = []
    
    # Get all image files
    image_files = glob.glob(os.path.join(images_dir, "*.jpg"))
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    # Create dictionaries mapping numbers to filenames
    image_dict = {}
    label_dict = {}
    
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        img_num = get_image_number(img_name)
        if img_num is not None:
            image_dict[img_num] = img_path
    
    for label_path in label_files:
        label_name = os.path.basename(label_path)
        label_num = get_label_number(label_name)
        if label_num is not None:
            label_dict[label_num] = label_path
    
    # Find matching pairs
    for num in image_dict.keys():
        if num in label_dict:
            matched_pairs.append((image_dict[num], label_dict[num]))
        else:
            print(f"Warning: No matching label for image number {num}")
    
    print(f"Found {len(matched_pairs)} matching image-label pairs")
    return matched_pairs

def split_dataset(frames_path, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split dataset into train, val, and test sets"""
    
    # Verify ratios sum to 1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, val, and test ratios must sum to 1.0")
    
    frames_path = Path(frames_path)
    
    # Create output directories
    create_directories(frames_path)
    
    # Process each camera view
    camera_views = ['back_left', 'back_right', 'front', 'top']
    
    for camera in camera_views:
        print(f"\nProcessing {camera} camera view...")
        
        images_dir = frames_path / 'images' / camera
        labels_dir = frames_path / 'labels' / camera
        
        # Match image and label files
        matched_pairs = match_files(images_dir, labels_dir)
        
        if not matched_pairs:
            print(f"No matched pairs found for {camera}")
            continue
        
        # Shuffle the pairs to ensure random distribution
        random.shuffle(matched_pairs)
        
        # Calculate split indices
        total_pairs = len(matched_pairs)
        train_end = int(total_pairs * train_ratio)
        val_end = train_end + int(total_pairs * val_ratio)
        
        # Split the data
        train_pairs = matched_pairs[:train_end]
        val_pairs = matched_pairs[train_end:val_end]
        test_pairs = matched_pairs[val_end:]
        
        print(f"  Train: {len(train_pairs)} pairs")
        print(f"  Val: {len(val_pairs)} pairs")
        print(f"  Test: {len(test_pairs)} pairs")
        
        # Copy files to respective directories
        splits = [
            ('train', train_pairs),
            ('val', val_pairs),
            ('test', test_pairs)
        ]
        
        for split_name, pairs in splits:
            for img_path, label_path in pairs:
                # Copy image
                img_dest = frames_path / split_name / 'images' / camera / os.path.basename(img_path)
                shutil.copy2(img_path, img_dest)
                
                # Copy label
                label_dest = frames_path / split_name / 'labels' / camera / os.path.basename(label_path)
                shutil.copy2(label_path, label_dest)
        
        print(f"  Completed {camera}")

def create_data_yaml(frames_path):
    """Create a data.yaml file for the split dataset"""
    frames_path = Path(frames_path)
    
    yaml_content = f"""# Dataset configuration for YOLO training
path: {frames_path.as_posix()}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images      # val images (relative to 'path')
test: test/images    # test images (relative to 'path')

# Classes
nc: 1  # number of classes
names: ['gas_bottle']  # class names
"""
    
    yaml_path = frames_path / 'data_split.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated data configuration file: {yaml_path}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define the frames directory path
    frames_dir = r"C:\Users\furqu\OneDrive\UCLL\Projects\Gassy\gass_GASSY\frames"
    
    print("Starting dataset split...")
    print(f"Source directory: {frames_dir}")
    print(f"Split ratios - Train: 70%, Val: 20%, Test: 10%")
    
    # Split the dataset
    split_dataset(frames_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)
    
    # Create data.yaml file
    create_data_yaml(frames_dir)
    
    print("\nDataset split completed successfully!")
    
    # Print summary
    frames_path = Path(frames_dir)
    for split in ['train', 'val', 'test']:
        total_images = 0
        total_labels = 0
        for camera in ['back_left', 'back_right', 'front', 'top']:
            img_count = len(list((frames_path / split / 'images' / camera).glob('*.jpg')))
            label_count = len(list((frames_path / split / 'labels' / camera).glob('*.txt')))
            total_images += img_count
            total_labels += label_count
        print(f"{split.capitalize()}: {total_images} images, {total_labels} labels")