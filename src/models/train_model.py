# src/models/train_model.py
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
import joblib
import os
import argparse  # NEW: For retraining flag
import logging  # NEW: For logging retraining
from datetime import datetime  # NEW: For tracking retraining time

# NEW: Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_features():
    """Load extracted features with validation split"""
    X_train = np.load(r'C:\Users\dell\Desktop\dyslexia\data\features\X_train.npy')
    y_train = np.load(r'C:\Users\dell\Desktop\dyslexia\data\features\y_train.npy')
    X_test = np.load(r'C:\Users\dell\Desktop\dyslexia\data\features\X_test.npy')
    y_test = np.load(r'C:\Users\dell\Desktop\dyslexia\data\features\y_test.npy')
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def evaluate_model(y_true, y_pred, y_proba=None):
    """Calculate classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
    
    return metrics

def train_random_forest(X_train, y_train, params=None):
    """Train Random Forest classifier"""
    params = params or {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, params=None):
    """Train SVM classifier"""
    params = params or {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'probability': True,
        'random_state': 42
    }
    model = SVC(**params)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_name):
    """Save trained model"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{model_name}.joblib')

# NEW: Add retraining validation function
def validate_retraining_data(X_train, y_train):
    """Validate data before retraining"""
    # Basic checks - add more as needed
    if len(X_train) == 0 or len(y_train) == 0:
        logger.error("Empty training data")
        return False
    if len(np.unique(y_train)) < 2:
        logger.error("Insufficient classes in training data")
        return False
    return True

# NEW: Main training function with retraining support
def main(retrain=False):
    """Main training function with retraining support"""
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_features()
        
        # For retraining, validate data first
        if retrain:
            logger.info(f"Starting retraining at {datetime.now()}")
            if not validate_retraining_data(X_train, y_train):
                logger.error("Data validation failed - skipping retraining")
                return
            
            logger.info("Data validation passed - proceeding with retraining")
        
        # Original training logic remains unchanged
        models = {
            'random_forest': train_random_forest(X_train, y_train),
            'svm': train_svm(X_train, y_train),
            # 'mlp': train_mlp(X_train, y_train)
        }
        
        best_model = None
        best_score = 0
        
        with mlflow.start_run():
            for name, model in models.items():
                # Validation evaluation
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                metrics = evaluate_model(y_val, y_pred, y_proba)
                
                # Log metrics
                mlflow.log_metrics({f'{name}_{k}': v for k, v in metrics.items()})
                mlflow.log_param(f'{name}_model', name)
                
                # Track best model
                if metrics['f1'] > best_score:
                    best_score = metrics['f1']
                    best_model = model
                    best_model_name = name
            
            # Save best model
            if best_model:
                save_model(best_model, 'best_model')
                mlflow.sklearn.log_model(best_model, best_model_name)
                logger.info(f"Best model: {best_model_name} with F1: {best_score:.4f}")
                
                # Test set evaluation
                y_test_pred = best_model.predict(X_test)
                test_metrics = evaluate_model(y_test, y_test_pred)
                mlflow.log_metrics({f'best_model_test_{k}': v for k, v in test_metrics.items()})
                
        return best_model
    
    except Exception as e:
        logger.error(f"Error during {'retraining' if retrain else 'training'}: {str(e)}")
        raise

if __name__ == "__main__":
    # NEW: Add argument parsing for retraining
    parser = argparse.ArgumentParser()
    parser.add_argument('--retrain', action='store_true', help='Flag for automated retraining')
    args = parser.parse_args()
    
    # Run main function with retraining flag
    main(retrain=args.retrain)