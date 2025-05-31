from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
import cv2
import mlflow.pyfunc
import os
from pathlib import Path
import logging
import time
from joblib import load
from contextlib import asynccontextmanager
from typing import Dict, Any
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from prometheus_client import Counter, Histogram, Gauge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus Metrics
REQUEST_COUNT = Counter(
    'app_request_count',
    'Application Request Count',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Application Request Latency',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'app_prediction_count',
    'Prediction Count by Class',
    ['prediction_class']
)

MODEL_LOAD_STATUS = Gauge(
    'app_model_load_status',
    'Model Loading Status'
)

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
            MODEL_LOAD_STATUS.set(0)  # 0 = loading
            
            # Load feature extractor
            import tensorflow as tf
            feature_extractor = tf.keras.models.load_model('/models/feature_extractor.h5')
            logger.info("Feature extractor loaded successfully")
            
            # Load classifier
            classifier = load('/models/best_model.joblib')
            logger.info("Classifier loaded successfully")
            
            MODEL_LOAD_STATUS.set(1)  # 1 = loaded
            break
        except Exception as e:
            if attempt == retries - 1:
                logger.error(f"Failed to load models after {retries} attempts: {str(e)}")
                MODEL_LOAD_STATUS.set(2)  # 2 = error
                raise
            logger.warning(f"Attempt {attempt+1} failed. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    yield  # App runs here
    
    # Cleanup
    feature_extractor = None
    classifier = None
    MODEL_LOAD_STATUS.set(0)  # 0 = unloaded

app = FastAPI(
    title="Dyslexia Detection API",
    lifespan=lifespan
)

# Instrument app for Prometheus
Instrumentator().instrument(app).expose(app)

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

def check_data_drift(current_features: np.ndarray):
    """Check for data drift using Evidently AI"""
    try:
        current_df = pd.DataFrame(current_features.reshape(1, -1))
        reference = pd.read_csv('data/reference.csv')  # Update path as needed
        
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current_df)
        return report
    except Exception as e:
        logger.error(f"Drift check failed: {str(e)}")
        raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """API endpoint for prediction"""
    start_time = time.time()
    if feature_extractor is None or classifier is None:
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            http_status=503
        ).inc()
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
                if isinstance(output, dict):
                    proba = output.get('probabilities', [output.get('predictions', [0])])[0]
                    prediction = int(output.get('predictions', [0])[0])
                else:
                    proba = [float(output[0]), 1.0 - float(output[0])]
                    prediction = int(round(output[0]))

            logger.info(f"Raw probabilities: {proba}")
            logger.info(f"Prediction: {prediction}")

            # Track prediction class
            PREDICTION_COUNT.labels(
                prediction_class="Reversal" if prediction else "Normal"
            ).inc()

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

            result = {
                "prediction": prediction,
                "probability": float(class_probs["Reversal" if prediction else "Normal"]),
                "class_probabilities": class_probs
            }

            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/predict",
                http_status=200
            ).inc()

            return result

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            REQUEST_COUNT.labels(
                method="POST",
                endpoint="/predict",
                http_status=500
            ).inc()
            raise HTTPException(status_code=500, detail=f"Prediction processing failed: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/predict",
            http_status=500
        ).inc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        REQUEST_LATENCY.labels(
            method="POST",
            endpoint="/predict"
        ).observe(time.time() - start_time)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if feature_extractor and classifier else "unhealthy"
    return {
        "status": status,
        "models_loaded": bool(feature_extractor and classifier)
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/monitor/drift")
async def monitor_drift(file: UploadFile = File(...)):
    """Check for data drift"""
    start_time = time.time()
    try:
        contents = await file.read()
        image = preprocess_image(contents)
        features = feature_extractor.predict(image)
        report = check_data_drift(features.flatten())
        
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/monitor/drift",
            http_status=200
        ).inc()
        
        return report.as_dict()
    except Exception as e:
        REQUEST_COUNT.labels(
            method="POST",
            endpoint="/monitor/drift",
            http_status=500
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(
            method="POST",
            endpoint="/monitor/drift"
        ).observe(time.time() - start_time)

# Your existing HTML response remains unchanged below
@app.get("/", response_class=HTMLResponse)
async def main():
    return HTMLResponse(content="""
    <html>
        <!-- Your existing HTML/JS/CSS remains exactly the same -->
    </html>
    """)
    

@app.get("/", response_class=HTMLResponse)
async def main():
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Dyslexia Detection</title>
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
            <style>
                :root {
                    --primary: #4361ee;
                    --secondary: #3f37c9;
                    --light: #f8f9fa;
                    --dark: #212529;
                    --success: #4cc9f0;
                    --danger: #f72585;
                }
                
                * {
                    box-sizing: border-box;
                    margin: 0;
                    padding: 0;
                }
                
                body {
                    font-family: 'Poppins', sans-serif;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    min-height: 100vh;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 20px;
                }
                
                .container {
                    max-width: 800px;
                    width: 100%;
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                    animation: fadeIn 0.5s ease-out;
                }
                
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                
                .header {
                    background: var(--primary);
                    color: white;
                    padding: 25px;
                    text-align: center;
                }
                
                .header h1 {
                    font-weight: 600;
                    font-size: 28px;
                    margin-bottom: 5px;
                }
                
                .header p {
                    opacity: 0.9;
                    font-weight: 300;
                }
                
                .content {
                    padding: 30px;
                }
                
                .upload-area {
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                    padding: 30px;
                    text-align: center;
                    margin-bottom: 25px;
                    transition: all 0.3s;
                    cursor: pointer;
                }
                
                .upload-area:hover {
                    border-color: var(--primary);
                    background: rgba(67, 97, 238, 0.05);
                }
                
                .upload-area.highlight {
                    border-color: var(--primary);
                    background: rgba(67, 97, 238, 0.05);
                }
                
                .upload-icon {
                    font-size: 48px;
                    color: var(--primary);
                    margin-bottom: 15px;
                }
                
                .btn {
                    background: var(--primary);
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.3s;
                    display: inline-block;
                    margin-top: 15px;
                }
                
                .btn:hover {
                    background: var(--secondary);
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                }
                
                .btn:disabled {
                    background: #ccc;
                    cursor: not-allowed;
                    transform: none;
                    box-shadow: none;
                }
                
                #fileInput {
                    display: none;
                }
                
                .result {
                    margin-top: 30px;
                    padding: 20px;
                    border-radius: 8px;
                    background: var(--light);
                    display: none;
                    animation: fadeIn 0.5s ease-out;
                }
                
                .result.show {
                    display: block;
                }
                
                .result h3 {
                    color: var(--dark);
                    margin-bottom: 15px;
                    font-size: 20px;
                }
                
                .result-item {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }
                
                .result-item:last-child {
                    border-bottom: none;
                }
                
                .progress-container {
                    width: 100%;
                    height: 8px;
                    background: #e9ecef;
                    border-radius: 4px;
                    margin-top: 5px;
                    overflow: hidden;
                }
                
                .progress-bar {
                    height: 100%;
                    border-radius: 4px;
                    transition: width 0.5s ease-out;
                }
                
                .normal-progress {
                    background: var(--success);
                }
                
                .reversal-progress {
                    background: var(--danger);
                }
                
                .error {
                    background: #fff5f5;
                    color: var(--danger);
                    border-left: 4px solid var(--danger);
                }
                
                .loading {
                    display: none;
                    text-align: center;
                    margin: 20px 0;
                }
                
                .spinner {
                    border: 4px solid rgba(0, 0, 0, 0.1);
                    border-radius: 50%;
                    border-top: 4px solid var(--primary);
                    width: 30px;
                    height: 30px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 10px;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .file-info {
                    margin-top: 15px;
                    font-size: 14px;
                    color: #666;
                }
                
                footer {
                    text-align: center;
                    margin-top: 30px;
                    font-size: 12px;
                    color: #666;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Dyslexia Detection System</h1>
                    <p>Upload a handwriting sample to analyze for dyslexia indicators</p>
                </div>
                
                <div class="content">
                    <div class="upload-area" id="uploadArea">
                        <div class="upload-icon">📁</div>
                        <h3>Drag & Drop your image here</h3>
                        <p>or click to browse files</p>
                        <input type="file" id="fileInput" name="file" accept="image/*" required>
                        <button class="btn" id="analyzeBtn" disabled>Analyze</button>
                    </div>
                    
                    <div class="file-info" id="fileInfo"></div>
                    
                    <div class="loading" id="loading">
                        <div class="spinner"></div>
                        <p>Analyzing your image...</p>
                    </div>
                    
                    <div class="result" id="result">
                        <h3>Analysis Results</h3>
                        <div id="resultContent"></div>
                    </div>
                </div>
                
                <footer>
                    Dyslexia Detection System &copy; 2023
                </footer>
            </div>
            
            <script>
                const uploadArea = document.getElementById('uploadArea');
                const fileInput = document.getElementById('fileInput');
                const analyzeBtn = document.getElementById('analyzeBtn');
                const fileInfo = document.getElementById('fileInfo');
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                // Handle drag and drop
                uploadArea.addEventListener('click', () => fileInput.click());
                
                fileInput.addEventListener('change', (e) => {
                    if (e.target.files.length) {
                        const file = e.target.files[0];
                        fileInfo.innerHTML = `Selected: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(2)} KB)`;
                        analyzeBtn.disabled = false;
                    }
                });
                
                uploadArea.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadArea.classList.add('highlight');
                });
                
                uploadArea.addEventListener('dragleave', () => {
                    uploadArea.classList.remove('highlight');
                });
                
                uploadArea.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadArea.classList.remove('highlight');
                    
                    if (e.dataTransfer.files.length) {
                        fileInput.files = e.dataTransfer.files;
                        const file = e.dataTransfer.files[0];
                        fileInfo.innerHTML = `Selected: <strong>${file.name}</strong> (${(file.size / 1024).toFixed(2)} KB)`;
                        analyzeBtn.disabled = false;
                    }
                });
                
                analyzeBtn.addEventListener('click', predict);
                
                async function predict() {
                    if (!fileInput.files.length) return;
                    
                    const formData = new FormData();
                    formData.append('file', fileInput.files[0]);
                    
                    try {
                        // Show loading state
                        loading.style.display = 'block';
                        result.classList.remove('show');
                        result.classList.remove('error');
                        
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error(`Error: ${response.statusText}`);
                        }
                        
                        const data = await response.json();
                        
                        // Hide loading, show results
                        loading.style.display = 'none';
                        
                        // Safe property access
                        const prediction = data.prediction ?? 0;
                        const probs = data.class_probabilities ?? { Normal: 0.5, Reversal: 0.5 };
                        const confidence = data.probability ?? (prediction ? probs.Reversal : probs.Normal);
                        
                        resultContent.innerHTML = `
                            <div class="result-item">
                                <span>Prediction:</span>
                                <strong style="color: ${prediction ? 'var(--danger)' : 'var(--success)'}">
                                    ${prediction ? 'Reversal Detected' : 'Normal'}
                                </strong>
                            </div>
                            
                            <div class="result-item">
                                <span>Confidence:</span>
                                <span>${(confidence * 100).toFixed(2)}%</span>
                            </div>
                            
                            <div class="result-item">
                                <span>Normal Probability:</span>
                                <span>${(probs.Normal * 100).toFixed(2)}%</span>
                            </div>
                            <div class="progress-container">
                                <div class="progress-bar normal-progress" style="width: ${probs.Normal * 100}%"></div>
                            </div>
                            
                            <div class="result-item">
                                <span>Reversal Probability:</span>
                                <span>${(probs.Reversal * 100).toFixed(2)}%</span>
                            </div>
                            <div class="progress-container">
                                <div class="progress-bar reversal-progress" style="width: ${probs.Reversal * 100}%"></div>
                            </div>
                        `;
                        
                        result.classList.add('show');
                    } catch (error) {
                        loading.style.display = 'none';
                        resultContent.innerHTML = `
                            <div style="color: var(--danger)">
                                <strong>Error:</strong> ${error.message}
                            </div>
                        `;
                        result.classList.add('show');
                        result.classList.add('error');
                    }
                }
            </script>
        </body>
    </html>
    """)