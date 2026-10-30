import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm
import glob
import gc

def load_papilledema_dataset(data_path='data', img_size=(128, 128), max_samples_per_class=1000):
    """
    Load Papilledema dataset with three classes: Normal, Papilledema, Pseudopapilledema
    """
    
    print("Loading Papilledema dataset...")
    
    base_path = data_path
    if not os.path.exists(base_path):
        print(f"Error: Dataset path {base_path} does not exist!")
        return None, None
    
    # Use smaller image size to save memory
    if img_size[0] > 128:
        img_size = (128, 128)
        print(f"Using reduced image size: {img_size} to save memory")
    
    images = []
    labels = []
    
    # Define the three classes
    classes = ['Normal', 'Papilledema', 'Pseudopapilledema']
    
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class {class_name} not found at {class_path}")
            continue
            
        print(f"\nProcessing {class_name} class...")
        
        class_images, class_labels = load_images_from_folder(class_path, class_name, img_size, max_samples_per_class)
        images.extend(class_images)
        labels.extend(class_labels)
        gc.collect()  # Free memory after each class
    
    if len(images) == 0:
        print("Error: No images were loaded!")
        return None, None
    
    X = np.array(images)
    y = np.array(labels)
    
    print(f"\nPapilledema dataset loaded successfully!")
    print(f"Total images: {X.shape[0]}")
    print_class_distribution(y)
    
    return X, y

def load_images_from_folder(folder_path, label, img_size, max_samples):
    """Load images from a specific folder with memory optimization"""
    images = []
    labels = []
    
    # Find all image files
    image_files = glob.glob(os.path.join(folder_path, '*.*'))
    image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif', '.tif', '.tiff', '.bmp'))]
    
    if max_samples:
        image_files = image_files[:max_samples]
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    for img_path in tqdm(image_files, desc=f"Loading {label}"):
        try:
            img = load_fundus_image(img_path, img_size)
            if img is not None:
                images.append(img)
                labels.append(label)
                
                # Free memory periodically
                if len(images) % 100 == 0:
                    gc.collect()
                    
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    return images, labels

def load_fundus_image(img_path, img_size):
    """Load and preprocess fundus image with memory optimization"""
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Resize to smaller dimensions
    img = cv2.resize(img, img_size)
    
    # Normalize to [0, 1] range
    img = img.astype(np.float32) / 255.0
    
    return img

def preprocess_medical_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Preprocess medical data for multi-class classification
    """
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Classes: {le.classes_}")
    print(f"Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )
    
    # Free original data
    del X, y_encoded
    gc.collect()
    
    # Second split: separate validation from train
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, 
        random_state=random_state, stratify=y_train_val
    )
    
    # Free intermediate arrays
    del X_train_val, y_train_val
    gc.collect()
    
    print(f"\nData Split:")
    print(f"  Train:      {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test:       {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, le

def print_class_distribution(y):
    """Print class distribution"""
    unique, counts = np.unique(y, return_counts=True)
    
    print("\nClass Distribution:")
    for class_name, count in zip(unique, counts):
        print(f"  {class_name}: {count} samples")
    
    # Calculate balance ratios
    if len(unique) >= 2:
        total = sum(counts)
        print(f"\nClass Percentages:")
        for class_name, count in zip(unique, counts):
            percentage = (count / total) * 100
            print(f"  {class_name}: {percentage:.1f}%")

# Keep the original functions for backward compatibility
def load_diabetic_retinopathy_binary(data_path='data', img_size=(128, 128), max_samples_per_class=1000, use_retino=False):
    """
    Original function kept for backward compatibility
    """
    print("Note: Using Papilledema dataset instead of Diabetic Retinopathy")
    return load_papilledema_dataset(data_path, img_size, max_samples_per_class)

def combine_datasets_smart(data_path='data', img_size=(128, 128), max_samples_per_class=800):
    """
    For Papilledema, we just use the single dataset
    """
    print("Loading Papilledema dataset...")
    return load_papilledema_dataset(data_path, img_size, max_samples_per_class)

def preprocess_medical_data_binary(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Binary version for backward compatibility
    """
    return preprocess_medical_data(X, y, test_size, val_size, random_state)

def print_class_distribution_binary(y):
    """Binary version for backward compatibility"""
    print_class_distribution(y)
# Keep the original functions for backward compatibility
def load_diabetic_retinopathy_binary(data_path='data', img_size=(128, 128), max_samples_per_class=1000, use_retino=False):
    """
    Original function kept for backward compatibility
    """
    print("Note: Using Papilledema dataset instead of Diabetic Retinopathy")
    return load_papilledema_dataset(data_path, img_size, max_samples_per_class)

def combine_datasets_smart(data_path='data', img_size=(128, 128), max_samples_per_class=800):
    """
    For Papilledema, we just use the single dataset
    """
    print("Loading Papilledema dataset...")
    return load_papilledema_dataset(data_path, img_size, max_samples_per_class)

def preprocess_medical_data_binary(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """
    Binary version for backward compatibility - now handles multi-class
    """
    return preprocess_medical_data(X, y, test_size, val_size, random_state)

def print_class_distribution_binary(y):
    """Binary version for backward compatibility"""
    print_class_distribution(y)