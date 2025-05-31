import numpy as np
import cv2
import logging
from typing import Dict, Any, Optional
from joblib import load
import tensorflow as tf
from fastapi import HTTPException
import time

logger = logging.getLogger(__name__)

class DyslexiaPredictor:
    def __init__(self, feature_extractor_path: str, classifier_path: str):
        self.feature_extractor: Optional[tf.keras.Model] = None
        self.classifier: Optional[Any] = None
        self.feature_extractor_path = feature_extractor_path
        self.classifier_path = classifier_path
        self._load_models()

    def _load_models(self) -> None:
        """Load ML models with retry logic"""
        retries = 3
        delay = 5
        
        for attempt in range(retries):
            try:
                logger.info("Loading models...")
                self.feature_extractor = tf.keras.models.load_model(self.feature_extractor_path)
                self.classifier = load(self.classifier_path)
                logger.info("Models loaded successfully")
                return
            except FileNotFoundError as e:
                logger.error(f"Model files not found: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail="Model files not found - please check paths"
                )
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
            nparr = np.frombuffer(image_bytes, np.uint8)
            if len(nparr) == 0:
                raise ValueError("Empty image data")
                
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Invalid image format")
                
            return np.expand_dims(cv2.resize(image, (128, 128)), axis=(0, -1)) / 255.0
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Make prediction on image bytes"""
        if self.feature_extractor is None or self.classifier is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded"
            )

        try:
            image = self.preprocess_image(image_bytes)
            features = self.feature_extractor.predict(image, verbose=0)
            
            if hasattr(self.classifier, 'predict_proba'):
                proba = self.classifier.predict_proba(features)[0]
                prediction = int(self.classifier.predict(features)[0])
            else:
                output = self.classifier.predict(features)
                if isinstance(output, (list, np.ndarray)) and len(output) > 0:
                    output = output[0]
                proba = [float(1 - output), float(output)]
                prediction = int(round(output))

            proba = np.clip(proba, 0, 1)
            if len(proba) < 2:
                proba = [1 - proba[0], proba[0]]

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
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )