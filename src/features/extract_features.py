import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten
import numpy as np
import mlflow
import os

def build_feature_extractor(input_shape=(128, 128, 1)):
    """
    Build CNN feature extractor with 4 conv layers and 2 max-pooling layers
    """
    inputs = Input(shape=input_shape)
    
    # Conv Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Conv Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Flatten features
    x = Flatten()(x)
    
    # Create model
    model = Model(inputs, x, name='feature_extractor')
    
    return model

def extract_features(X, model):
    """Extract features using the CNN model"""
    X_processed = X / 255.0  # Normalize pixel values
    features = model.predict(X_processed)
    return features

def save_features(features, labels, split='train'):
    """Save extracted features and labels to disk"""
    os.makedirs('data/features', exist_ok=True)
    np.save(f'data/features/X_{split}.npy', features)
    np.save(f'data/features/y_{split}.npy', labels)

if __name__ == '__main__':
    # Load data
    X_train = np.load('data/processed/X_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Build and save feature extractor model
    feature_extractor = build_feature_extractor()
    feature_extractor.summary()
    feature_extractor.save('models/feature_extractor.h5')
    
    # Set MLflow experiment (creates it if it doesn't exist)
    mlflow.set_experiment("Dyslexia Detection")
    
    with mlflow.start_run():
        # Log the model
        mlflow.tensorflow.log_model(feature_extractor, "feature_extractor")
        
        # Extract features for each dataset split
        print("Extracting training features...")
        train_features = extract_features(X_train, feature_extractor)
        print("Extracting validation features...")
        val_features = extract_features(X_val, feature_extractor)
        print("Extracting test features...")
        test_features = extract_features(X_test, feature_extractor)
        
        # Save features and labels
        print("Saving features...")
        save_features(train_features, y_train, 'train')
        save_features(val_features, y_val, 'val')
        save_features(test_features, y_test, 'test')
        
        # Log shapes of extracted features
        mlflow.log_params({
            'train_features_shape': train_features.shape,
            'val_features_shape': val_features.shape,
            'test_features_shape': test_features.shape
        })
    
    print("Feature extraction complete!")
