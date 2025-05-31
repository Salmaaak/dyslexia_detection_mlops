from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
#from fastapi.staticfiles import StaticFiles
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
    """Lifespan handler for model loading and cleanup"""
    global feature_extractor, classifier
    
    # Load models
    retries = 3
    delay = 5
    
    for attempt in range(retries):
        try:
            logger.info("Loading models...")
            
            # Load feature extractor
            import tensorflow as tf
            feature_extractor = tf.keras.models.load_model('/models/feature_extractor.h5')
            logger.info("Feature extractor loaded successfully")
            
            # Load classifier
            classifier = load('/models/best_model.joblib')
            logger.info("Classifier loaded successfully")
            
            break
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to load models after {retries} attempts: {str(e)}")
                raise
            logger.warning(f"Attempt {attempt+1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    yield  # App runs here
    
    # Cleanup
    feature_extractor = None
    classifier = None

app = FastAPI(
    title="Dyslexia Detection API",
    lifespan=lifespan
)

# Mount static files from /app/static directory
#app.mount("/static", StaticFiles(directory="/app/static"), name="static")

def preprocess_image(image_bytes):
    """Convert uploaded file to processed image array"""
    try:
        # Read image
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Resize to match feature extractor input size
        image = cv2.resize(image, (128, 128))
        
        # Normalize and expand dimensions
        image = np.expand_dims(image, axis=(0, -1)) / 255.0
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.post("/predict")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint for prediction"""
    if feature_extractor is None or classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Read and preprocess image
        contents = await file.read()
        image = preprocess_image(contents)

        # Extract features
        logger.info("Extracting features...")
        features = feature_extractor.predict(image)
        logger.info(f"Features shape: {features.shape}")

        # Make prediction
        logger.info("Making prediction...")
        try:
            if hasattr(classifier, 'predict_proba'):
                logger.info("Using predict_proba")
                proba = classifier.predict_proba(features)[0]
                prediction = int(classifier.predict(features)[0])
            else:
                logger.info("Using direct prediction")
                output = classifier.predict(features)
                # Handle different classifier output formats
                if isinstance(output, dict):
                    proba = output.get('probabilities', [output.get('predictions', [0])])[0]
                    prediction = int(output.get('predictions', [0])[0])
                else:
                    proba = [float(output[0]), 1.0 - float(output[0])]
                    prediction = int(round(output[0]))

            logger.info(f"Raw probabilities: {proba}")
            logger.info(f"Prediction: {prediction}")

            # Ensure we always return both classes
            if len(proba) == 2:  # Normal and Reversal
                class_probs = {
                    "Normal": float(proba[0]),
                    "Reversal": float(proba[1])
                }
            else:  # Handle unexpected cases
                class_probs = {
                    "Normal": float(proba[0]),
                    "Reversal": 1.0 - float(proba[0])
                }

            return {
                "prediction": prediction,
                "probability": float(class_probs["Reversal" if prediction else "Normal"]),
                "class_probabilities": class_probs
            }

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Prediction processing failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")    

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if feature_extractor and classifier else "unhealthy",
        "models_loaded": bool(feature_extractor and classifier)
    }

@app.get("/", response_class=HTMLResponse)
async def main():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Dyslexia Detection</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #2c3e50; }
                form { margin: 20px 0; }
                .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <h1>Dyslexia Detection System</h1>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" id="fileInput" name="file" accept="image/*" required>
                <button type="button" onclick="predict()">Analyze</button>
            </form>
            <div id="result" class="result" style="display: none;"></div>
            
            <script>
                async function predict() {
                    const fileInput = document.getElementById('fileInput');
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = '';
                    resultDiv.classList.remove('error');
                    
                    if (!fileInput.files.length) {
                        showError('Please select an image file');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    
                    try {
                        resultDiv.innerHTML = 'Processing...';
                        resultDiv.style.display = 'block';
                        
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const data = await response.json();
                        
                        // Safe property access
                        const prediction = data.prediction ?? 0;
                        const probs = data.class_probabilities ?? { Normal: 0, Reversal: 0 };
                        const confidence = data.probability ?? (prediction ? probs.Reversal : probs.Normal);
                        
                        resultDiv.innerHTML = `
                            <h3>Results:</h3>
                            <p>Prediction: ${prediction ? 'Reversal' : 'Normal'}</p>
                            <p>Confidence: ${(confidence * 100).toFixed(2)}%</p>
                            <p>Normal Probability: ${(probs.Normal * 100).toFixed(2)}%</p>
                            <p>Reversal Probability: ${(probs.Reversal * 100).toFixed(2)}%</p>
                        `;
                    } catch (error) {
                        showError(error.message);
                    }
                }
                
                function showError(message) {
                    const resultDiv = document.getElementById('result');
                    resultDiv.innerHTML = `Error: ${message}`;
                    resultDiv.style.display = 'block';
                    resultDiv.classList.add('error');
                }
            </script>
        </body>
    </html>
    """)