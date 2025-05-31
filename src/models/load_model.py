import mlflow.pyfunc
import joblib
from pathlib import Path
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_production_model():
    """Load the production model from MLflow model registry"""
    try:
        logger.info("Loading production model from MLflow...")
        model = mlflow.pyfunc.load_model(
            model_uri="models:/dyslexia-detector/Production"
        )
        logger.info("Successfully loaded production model from MLflow")
        return model
    except Exception as e:
        logger.error(f"Failed to load from MLflow: {str(e)}")
        raise

def load_local_model(model_path):
    """Load a model from local disk"""
    try:
        logger.info(f"Loading local model from {model_path}...")
        if model_path.suffix == '.joblib':
            model = joblib.load(model_path)
        elif model_path.suffix == '.h5':
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        logger.info("Successfully loaded local model")
        return model
    except Exception as e:
        logger.error(f"Failed to load local model: {str(e)}")
        raise

def get_latest_production_model():
    """Get the latest production model, falling back to local if needed"""
    try:
        return load_production_model()
    except Exception as e:
        logger.warning("Falling back to local model")
        model_path = Path('models/random_forest.joblib')
        if not model_path.exists():
            raise FileNotFoundError("No local model found for fallback")
        return load_local_model(model_path)