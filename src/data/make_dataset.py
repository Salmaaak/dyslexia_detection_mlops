import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import dvc.api

def load_data(data_path):
    # Load and split data
    normal_images = []
    reversal_images = []
    
    normal_path = os.path.join(data_path, 'Normal')
    reversal_path = os.path.join(data_path, 'Reversal')
    
    for img_file in os.listdir(normal_path):
        img = cv2.imread(os.path.join(normal_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        normal_images.append(img)
    
    for img_file in os.listdir(reversal_path):
        img = cv2.imread(os.path.join(reversal_path, img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        reversal_images.append(img)
    
    # Create labels
    X = np.array(normal_images + reversal_images)
    y = np.array([0]*len(normal_images) + [1]*len(reversal_images))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    with dvc.api.open('data/raw', mode='rb') as fd:
        X_train, X_test, y_train, y_test = load_data('data/raw')
        
    # Save processed data
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_test.npy', X_test)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)