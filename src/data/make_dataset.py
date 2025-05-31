import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import dvc.api

# Target image size
IMG_WIDTH = 128
IMG_HEIGHT = 128

def load_and_resize_image(img_path):
    """Load an image and resize it to target dimensions"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        return img
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        return None

def load_data(data_path):
    # Load and split data
    normal_images = []
    reversal_images = []
    
    normal_path = os.path.join(data_path, 'Normal')
    reversal_path = os.path.join(data_path, 'Reversal')
    
    # Load normal images
    for img_file in os.listdir(normal_path):
        img_path = os.path.join(normal_path, img_file)
        img = load_and_resize_image(img_path)
        if img is not None:
            normal_images.append(img)
    
    # Load reversal images
    for img_file in os.listdir(reversal_path):
        img_path = os.path.join(reversal_path, img_file)
        img = load_and_resize_image(img_path)
        if img is not None:
            reversal_images.append(img)
    
    # Verify we have images
    if not normal_images or not reversal_images:
        raise ValueError("No valid images found in one or both directories")
    
    # Create arrays with consistent shape
    X = np.array(normal_images + reversal_images)
    y = np.array([0]*len(normal_images) + [1]*len(reversal_images))
    
    # Add channel dimension (for CNN)
    X = np.expand_dims(X, axis=-1)
    
    # First split: train (80%) and temp (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Second split: validation (10%) and test (10%) of original
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    # Make sure the raw data is in data/raw with Normal and Reversal subfolders
    print("Loading and processing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data('data/raw')
    
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # Save processed data
    print("Saving processed data...")
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/y_test.npy', y_test)
    
    print("Data processing complete!")
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")