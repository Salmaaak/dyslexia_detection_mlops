import numpy as np
import cv2
import logging
from typing import Dict, Any
from joblib import load
import tensorflow as tf
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class DyslexiaPredictor:
    def __init__(self, feature_extractor_path: str, classifier_path: str):
        self.feature_extractor = None
        self.classifier = None
        self.load_models(feature_extractor_path, classifier_path)

    def load_models(self, feature_extractor_path: str, classifier_path: str):
        """Load ML models with retry logic"""
        retries = 3
        delay = 5
        
        for attempt in range(retries):
            try:
                logger.info("Loading models...")
                self.feature_extractor = tf.keras.models.load_model(feature_extractor_path)
                self.classifier = load(classifier_path)
                logger.info("Models loaded successfully")
                return
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Failed to load models after {retries} attempts: {str(e)}")
                    raise HTTPException(
                        status_code=503,
                        detail="Service unavailable - failed to load models"
                    )
                logger.warning(f"Attempt {attempt+1} failed. Retrying in {delay} seconds...")
                time.sleep(delay)

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert uploaded file to processed image array"""
        try:
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (128, 128))
            return np.expand_dims(image, axis=(0, -1)) / 255.0
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid image file")

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Make prediction on image bytes"""
        if not self.feature_extractor or not self.classifier:
            raise HTTPException(status_code=503, detail="Models not loaded")

        try:
            image = self.preprocess_image(image_bytes)
            features = self.feature_extractor.predict(image)
            
            if hasattr(self.classifier, 'predict_proba'):
                proba = self.classifier.predict_proba(features)[0]
                prediction = int(self.classifier.predict(features)[0])
            else:
                output = self.classifier.predict(features)
                if isinstance(output, dict):
                    proba = output.get('probabilities', [output.get('predictions', [0])])[0]
                    prediction = int(output.get('predictions', [0])[0])
                else:
                    proba = [float(output[0]), 1.0 - float(output[0])]
                    prediction = int(round(output[0]))

            return {
                "prediction": prediction,
                "probability": float(proba[1]),
                "class_probabilities": {
                    "Normal": float(proba[0]),
                    "Reversal": float(proba[1])
                }
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")