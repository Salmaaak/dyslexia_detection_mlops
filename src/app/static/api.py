from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import mlflow.pyfunc
import os
from pathlib import Path
import logging
import time
from joblib import load
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to hold models
feature_extractor = None
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for model loading"""
    global feature_extractor, classifier
    
    # Load models
    retries = 3
    delay = 5
    
    for attempt in range(retries):
        try:
            # Load feature extractor
            logger.info("Loading feature extractor...")
            import tensorflow as tf
            feature_extractor = tf.keras.models.load_model('models/feature_extractor.h5')
            logger.info("Feature extractor loaded successfully")
            
            # Load classifier
            classifier = load_classifier()
            logger.info("Classifier loaded successfully")
            break
        except Exception as e:
            if attempt == retries - 1:
                logger.error("Failed to load models after retries")
                raise
            logger.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    yield
    # Clean up if needed
    feature_extractor = None
    classifier = None

app = FastAPI(title="Dyslexia Detection API", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

def load_classifier():
    """Load classifier model with proper path handling"""
    try:
        # Try MLflow first
        # classifier = mlflow.pyfunc.load_model("models:/dyslexia-detector/Production")
        
        # Try local model - using container path
        model_path = Path("/models/best_model.joblib")  # Docker container path
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        return load(model_path)
    except Exception as e:
        logger.error(f"Failed to load classifier: {str(e)}")
        raise

def preprocess_image(image_bytes):
    """Convert uploaded file to processed image array"""
    try:
        # Read image
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Resize if needed (to match feature extractor input size)
        image = cv2.resize(image, (128, 128))
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint for prediction"""
    if feature_extractor is None or classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = preprocess_image(contents)
        
        # Extract features
        features = feature_extractor.predict(np.expand_dims(image, axis=(0, -1))) / 255.0
        
        # Make prediction
        if hasattr(classifier, 'predict'):
            # Standard sklearn interface
            prediction = classifier.predict(features)
            proba = classifier.predict_proba(features)[0]
        else:
            # MLflow pyfunc interface
            output = classifier.predict(features)
            prediction = output['predictions']
            proba = output['probabilities'][0]
        
        return {
            "prediction": int(prediction[0]),
            "probability": float(proba[prediction[0]]),
            "class_probabilities": {
                "Normal": float(proba[0]),
                "Reversal": float(proba[1])
            }
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = feature_extractor is not None and classifier is not None
    return {
        "status": "healthy" if status else "unhealthy",
        "models_loaded": status
    }

@app.get("/", response_class=HTMLResponse)
async def main():
    """Serve HTML interface"""
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)