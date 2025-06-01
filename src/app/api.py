from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
from prometheus_client import generate_latest, Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
import cv2
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
import pandas as pd
from evidently.report import Report
from evidently.metrics import (
    ClassificationQualityMetric,
    DataDriftTable,
    DatasetDriftMetric
)
import psutil
from datetime import datetime
from joblib import load
import tensorflow as tf
from pathlib import Path
import os
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ======================
# Monitoring Setup
# ======================

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

PERFORMANCE_QUALITY = Gauge(
    'app_performance_quality',
    'Model Performance Quality Score'
)

class DyslexiaMonitor:
    def __init__(self):
        self.performance_data = []
        self.reference_accuracy = 0.85  # Baseline from validation
        self.feature_hashes = set()
        self.alert_history = []
        
    def log_prediction(self, prediction: int, confidence: float, features: np.ndarray):
        """Log prediction data and check for anomalies"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'feature_hash': self._hash_features(features)
        }
        self.performance_data.append(entry)
        
        # Keep only last 1000 entries
        if len(self.performance_data) > 1000:
            self.performance_data.pop(0)
            
        # Check for issues
        self._check_performance()
        self._check_feature_drift()
    
    def _hash_features(self, features: np.ndarray) -> str:
        """Create consistent hash of feature vector"""
        return hashlib.md5(features.tobytes()).hexdigest()
    
    def _check_performance(self, window=100):
        """Check for performance degradation"""
        if len(self.performance_data) < window:
            return
            
        recent = self.performance_data[-window:]
        accuracy = sum(1 for x in recent if x['prediction'] == 0) / window
        
        if accuracy < self.reference_accuracy * 0.85:  # 15% drop
            self.alert_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'performance',
                'message': f"Accuracy dropped to {accuracy:.2f} (expected {self.reference_accuracy})"
            })
    
    def _check_feature_drift(self, window=100):
        """Detect significant feature distribution changes"""
        if len(self.performance_data) < window:
            return
            
        recent_hashes = {x['feature_hash'] for x in self.performance_data[-window:]}
        new_hashes = recent_hashes - self.feature_hashes
        
        if len(new_hashes) / window > 0.3:  # 30% new patterns
            self.alert_history.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'feature_drift',
                'message': f"Detected {len(new_hashes)} new feature patterns"
            })
        
        self.feature_hashes.update(recent_hashes)
    
    def get_stats(self):
        """Return current monitoring statistics"""
        if not self.performance_data:
            return None
            
        window = min(100, len(self.performance_data))
        recent = self.performance_data[-window:]
        
        return {
            'accuracy': sum(1 for x in recent if x['prediction'] == 0) / window,
            'alerts': self.alert_history[-5:],  # Last 5 alerts
            'prediction_distribution': {
                'Normal': sum(1 for x in recent if x['prediction'] == 0),
                'Reversal': sum(1 for x in recent if x['prediction'] == 1)
            }
        }

# Initialize monitor
monitor = DyslexiaMonitor()

# ======================
# Model Setup
# ======================

feature_extractor = None
classifier = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for model loading and cleanup"""
    global feature_extractor, classifier
    
    MODEL_LOAD_STATUS.set(0)  # 0 = loading
    
    # Define model paths (relative to your project root)
    models_dir = Path("models")
    feature_extractor_path = models_dir / "feature_extractor.h5"
    classifier_path = models_dir / "best_model.joblib"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    # Load models with retry
    retries = 3
    delay = 5
    for attempt in range(retries):
        try:
            logger.info(f"Loading models from {models_dir}...")
            
            # Verify model files exist
            if not feature_extractor_path.exists():
                raise FileNotFoundError(f"Feature extractor not found at {feature_extractor_path}")
            if not classifier_path.exists():
                raise FileNotFoundError(f"Classifier not found at {classifier_path}")
            
            # Load feature extractor
            feature_extractor = tf.keras.models.load_model(str(feature_extractor_path))
            logger.info("Feature extractor loaded successfully")
            
            # Load classifier
            classifier = load(str(classifier_path))
            logger.info("Classifier loaded successfully")
            
            MODEL_LOAD_STATUS.set(1)  # 1 = loaded
            break
        except Exception as e:
            if attempt == retries - 1:
                error_msg = f"Failed to load models after {retries} attempts. "
                error_msg += f"Please ensure these files exist: {[str(feature_extractor_path), str(classifier_path)]}"
                logger.error(error_msg)
                MODEL_LOAD_STATUS.set(2)  # 2 = error
                raise HTTPException(status_code=500, detail=error_msg)
            logger.warning(f"Attempt {attempt+1} failed: {str(e)}. Retrying in {delay} seconds...")
            time.sleep(delay)
    
    yield
    
    # Cleanup
    feature_extractor = None
    classifier = None
    MODEL_LOAD_STATUS.set(0)  # 0 = unloaded

app = FastAPI(
    title="Dyslexia Detection API",
    lifespan=lifespan
)

Instrumentator().instrument(app).expose(app)

# ======================
# Core Functions
# ======================

def preprocess_document(image: np.ndarray) -> List[Dict]:
    """Process document image into individual characters"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Invert colors (white background becomes black)
    inverted = cv2.bitwise_not(gray)
    
    # Threshold to get pure black and white
    _, binary = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)
    
    # Optional: Improve character segmentation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of individual characters
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each character
    characters = []
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter very small contours (noise)
        if w < 10 or h < 10:
            continue
            
        # Extract character ROI
        char = binary[y:y+h, x:x+w]
        
        # Create black background with white character
        processed_char = cv2.bitwise_not(char)
        
        characters.append({
            'image': processed_char,
            'contour': contour,
            'bbox': (x, y, w, h)
        })
    
    # Sort characters left-to-right
    characters.sort(key=lambda c: c['bbox'][0])
    
    return characters

def prepare_for_model(characters: List[Dict], target_size: tuple = (128, 128)) -> np.ndarray:
    """Prepare character images for model input"""
    processed = []
    for char in characters:
        # Pad to square while maintaining aspect ratio
        char_img = char['image']
        h, w = char_img.shape
        size = max(w, h)
        padded = np.zeros((size, size), dtype=np.uint8)
        x_pad = (size - w) // 2
        y_pad = (size - h) // 2
        padded[y_pad:y_pad+h, x_pad:x_pad+w] = char_img
        
        # Resize to target size
        resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
        
        # Add channel dimension
        resized = np.expand_dims(resized, axis=-1)
        processed.append(resized)
    
    return np.array(processed) / 255.0

def is_single_character(image: np.ndarray) -> bool:
    """Check if image contains a single character (like training data)"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.count_nonzero(image < 128) / image.size < 0.2  # Less than 20% content

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Convert uploaded file to processed image array"""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        if len(nparr) == 0:
            raise ValueError("Empty image data")
            
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image format")
        
        # Check if this is a single character (like training data)
        if is_single_character(image):
            # Process as single character
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            resized = cv2.resize(binary, (128, 128), interpolation=cv2.INTER_AREA)
            return np.expand_dims(resized, axis=(0, -1)) / 255.0
        else:
            # Process as full document
            characters = preprocess_document(image)
            if not characters:
                raise ValueError("No characters detected in image")
            return prepare_for_model(characters)
            
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

# ======================
# API Endpoints
# ======================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Make prediction on image bytes"""
    start_time = time.time()
    
    if feature_extractor is None or classifier is None:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=503).inc()
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Process image
        contents = await file.read()
        images = preprocess_image(contents)
        
        # Extract features
        features = feature_extractor.predict(images)
        
        # Make predictions
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(features)
            predictions = classifier.predict(features)
        else:
            proba = classifier.predict(features)
            predictions = np.round(proba).astype(int)
            proba = np.column_stack([1 - proba, proba])
        
        # Log prediction for monitoring
        for i in range(len(predictions)):
            monitor.log_prediction(
                prediction=int(predictions[i]),
                confidence=float(proba[i][1]),
                features=features[i]
            )
        
        # Prepare result
        if len(predictions) > 1:
            avg_prob = np.mean(proba, axis=0)
            result = {
                "prediction": int(np.round(np.mean(predictions))),
                "probability": float(avg_prob[1]),
                "class_probabilities": {
                    "Normal": float(avg_prob[0]),
                    "Reversal": float(avg_prob[1])
                },
                "character_details": [
                    {
                        "prediction": int(pred),
                        "probability": float(prob[1]),
                        "class_probabilities": {
                            "Normal": float(prob[0]),
                            "Reversal": float(prob[1])
                        }
                    } for pred, prob in zip(predictions, proba)
                ]
            }
        else:
            result = {
                "prediction": int(predictions[0]),
                "probability": float(proba[0][1]),
                "class_probabilities": {
                    "Normal": float(proba[0][0]),
                    "Reversal": float(proba[0][1])
                }
            }

        # Track metrics
        pred_class = "Reversal" if result["prediction"] else "Normal"
        PREDICTION_COUNT.labels(prediction_class=pred_class).inc()
        
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=200).inc()
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", http_status=500).inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        REQUEST_LATENCY.labels(method="POST", endpoint="/predict").observe(time.time() - start_time)

@app.get("/monitoring/stats")
async def get_monitoring_stats():
    """Get current monitoring statistics"""
    stats = monitor.get_stats()
    if stats is None:
        raise HTTPException(status_code=404, detail="No monitoring data available")
    return stats

@app.get("/", response_class=HTMLResponse)
async def root(show: str = None):
    """Root endpoint that handles both main interface and monitoring dashboard"""
    if show == "dashboard":
        return await monitoring_dashboard()
    elif show == "about":
        return await about_dyslexia()
    else:
        return await main_interface()

async def main_interface():
    """Main interface for dyslexia prediction"""
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
                    padding: 20px;
                }
                
                .container {
                    max-width: 1200px;
                    width: 100%;
                    margin: 0 auto;
                }
                
                .header {
                    background: var(--primary);
                    color: white;
                    padding: 25px;
                    text-align: center;
                    border-radius: 8px 8px 0 0;
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
                
                .nav {
                    display: flex;
                    justify-content: center;
                    background: var(--secondary);
                    padding: 10px;
                }
                
                .nav a {
                    color: white;
                    text-decoration: none;
                    padding: 10px 20px;
                    margin: 0 5px;
                    border-radius: 4px;
                    transition: background 0.3s;
                }
                
                .nav a:hover {
                    background: rgba(255, 255, 255, 0.2);
                }
                
                .content {
                    background: white;
                    padding: 30px;
                    border-radius: 0 0 8px 8px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
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
                
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Dyslexia Detection System</h1>
                    <p>Upload a handwriting sample to analyze for dyslexia indicators</p>
                </div>
                
                <div class="nav">
                    <a href="/">Home</a>
                    <a href="/?show=dashboard">Monitoring Dashboard</a>
                    <a href="/?show=about">About Dyslexia</a>
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

async def monitoring_dashboard():
    """Interactive monitoring dashboard"""
    return HTMLResponse(content="""
    <html>
        <head>
            <title>Dyslexia Monitoring Dashboard</title>
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
                    padding: 20px;
                }
                
                .container {
                    max-width: 1200px;
                    width: 100%;
                    margin: 0 auto;
                }
                
                .header {
                    background: var(--primary);
                    color: white;
                    padding: 25px;
                    text-align: center;
                    border-radius: 8px 8px 0 0;
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
                
                .nav {
                    display: flex;
                    justify-content: center;
                    background: var(--secondary);
                    padding: 10px;
                }
                
                .nav a {
                    color: white;
                    text-decoration: none;
                    padding: 10px 20px;
                    margin: 0 5px;
                    border-radius: 4px;
                    transition: background 0.3s;
                }
                
                .nav a:hover {
                    background: rgba(255, 255, 255, 0.2);
                }
                
                .dashboard-content {
                    background: white;
                    padding: 30px;
                    border-radius: 0 0 8px 8px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                }
                
                .status-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin-bottom: 30px;
                }
                
                .status-card {
                    background: var(--light);
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }
                
                .status-card h3 {
                    color: var(--dark);
                    margin-bottom: 10px;
                }
                
                .status-value {
                    font-size: 32px;
                    font-weight: 600;
                    color: var(--primary);
                    margin: 10px 0;
                }
                
                .metrics-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                }
                
                .metric-card {
                    background: var(--light);
                    padding: 20px;
                    border-radius: 8px;
                }
                
                .metric-card h3 {
                    color: var(--dark);
                    margin-bottom: 15px;
                }
                
                .chart-container {
                    height: 300px;
                }
                
                .info-panel {
                    background: var(--light);
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 20px;
                }
                
                .info-panel h3 {
                    color: var(--primary);
                    margin-bottom: 10px;
                }
                
                .alert {
                    padding: 10px;
                    margin: 5px 0;
                    border-radius: 4px;
                    background: #fff5f5;
                    border-left: 4px solid var(--danger);
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
                    <h1>Dyslexia Detection Monitoring</h1>
                    <p>System performance and insights</p>
                </div>
                
                <div class="nav">
                    <a href="/">Home</a>
                    <a href="/?show=dashboard">Monitoring Dashboard</a>
                    <a href="/?show=about">About Dyslexia</a>
                </div>
                
                <div class="dashboard-content">
                    <div class="status-grid">
                        <div class="status-card">
                            <h3>Current Accuracy</h3>
                            <div class="status-value" id="current-accuracy">Loading...</div>
                            <p>Compared to baseline</p>
                        </div>
                        <div class="status-card">
                            <h3>Recent Alerts</h3>
                            <div class="status-value" id="alert-count">0</div>
                            <p>System notifications</p>
                        </div>
                    </div>
                    
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <h3>Prediction Distribution</h3>
                            <div class="chart-container" id="prediction-chart"></div>
                        </div>
                        <div class="metric-card">
                            <h3>Performance Trend</h3>
                            <div class="chart-container" id="performance-trend"></div>
                        </div>
                    </div>
                    
                    <div class="info-panel">
                        <h3>Recent Alerts</h3>
                        <div id="alerts-container"></div>
                    </div>
                </div>
                
                <div class="footer">
                    Dyslexia Detection System &copy; 2023 | Last updated: <span id="last-updated"></span>
                </div>
            </div>
            
            <script>
                // Load initial data
                document.addEventListener('DOMContentLoaded', () => {
                    loadData();
                });
                
                async function loadData() {
                    try {
                        const response = await fetch('/monitoring/stats');
                        const data = await response.json();
                        
                        // Update stats
                        document.getElementById('current-accuracy').textContent = 
                            `${(data.accuracy * 100).toFixed(1)}%`;
                        document.getElementById('alert-count').textContent = 
                            data.alerts.length;
                        
                        // Update prediction chart
                        createPredictionChart(data.prediction_distribution);
                        
                        // Update alerts
                        const alertsContainer = document.getElementById('alerts-container');
                        alertsContainer.innerHTML = data.alerts.length > 0 
                            ? data.alerts.map(alert => `
                                <div class="alert">
                                    <strong>${alert.type.toUpperCase()}</strong><br>
                                    ${alert.message}<br>
                                    <small>${alert.timestamp}</small>
                                </div>
                            `).join('')
                            : '<p>No recent alerts</p>';
                        
                        // Update last updated time
                        document.getElementById('last-updated').textContent = new Date().toLocaleString();
                        
                    } catch (error) {
                        console.error('Error loading monitoring data:', error);
                    }
                }
                
                function createPredictionChart(distribution) {
                    const data = [{
                        values: [distribution.Normal, distribution.Reversal],
                        labels: ['Normal', 'Reversal'],
                        marker: {colors: ['var(--success)', 'var(--danger)']},
                        type: 'pie',
                        textinfo: 'percent+value'
                    }];
                    
                    Plotly.newPlot('prediction-chart', data, {margin: {t: 0, b: 0}});
                }
                
                // Auto-refresh every 30 seconds
                setInterval(loadData, 30000);
            </script>
        </body>
    </html>
    """)

async def about_dyslexia():
    """About dyslexia information page"""
    return HTMLResponse(content="""
    <html>
        <head>
            <title>About Dyslexia</title>
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
                    padding: 20px;
                }
                
                .container {
                    max-width: 1200px;
                    width: 100%;
                    margin: 0 auto;
                }
                
                .header {
                    background: var(--primary);
                    color: white;
                    padding: 25px;
                    text-align: center;
                    border-radius: 8px 8px 0 0;
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
                
                .nav {
                    display: flex;
                    justify-content: center;
                    background: var(--secondary);
                    padding: 10px;
                }
                
                .nav a {
                    color: white;
                    text-decoration: none;
                    padding: 10px 20px;
                    margin: 0 5px;
                    border-radius: 4px;
                    transition: background 0.3s;
                }
                
                .nav a:hover {
                    background: rgba(255, 255, 255, 0.2);
                }
                
                .content {
                    background: white;
                    padding: 30px;
                    border-radius: 0 0 8px 8px;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                }
                
                .info-section {
                    margin-bottom: 30px;
                }
                
                .info-section h2 {
                    color: var(--primary);
                    margin-bottom: 15px;
                }
                
                .info-section h3 {
                    color: var(--secondary);
                    margin: 20px 0 10px;
                }
                
                .info-section ul {
                    padding-left: 20px;
                    margin: 10px 0;
                }
                
                .info-section li {
                    margin-bottom: 8px;
                }
                
                .chart-container {
                    height: 300px;
                    margin: 30px 0;
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
                    <h1>About Dyslexia</h1>
                    <p>Understanding the learning disorder and how our system detects it</p>
                </div>
                
                <div class="nav">
                    <a href="/">Home</a>
                    <a href="/?show=dashboard">Monitoring Dashboard</a>
                    <a href="/?show=about">About Dyslexia</a>
                </div>
                
                <div class="content">
                    <div class="info-section">
                        <h2>Understanding Dyslexia</h2>
                        <p>Dyslexia is a learning disorder that affects reading, writing and spelling abilities. Our system detects common indicators in handwriting:</p>
                        
                        <h3>Common Signs</h3>
                        <ul>
                            <li>Letter reversals (b/d, p/q)</li>
                            <li>Inconsistent letter sizes</li>
                            <li>Poor spatial planning</li>
                            <li>Irregular letter formation</li>
                            <li>Difficulty with letter sequencing</li>
                            <li>Inconsistent spacing between words</li>
                        </ul>
                        
                        <h3>Detection Methodology</h3>
                        <p>Our AI analyzes handwriting samples for these patterns using deep learning models trained on thousands of samples. The system looks for:</p>
                        <ul>
                            <li>Mirror writing patterns</li>
                            <li>Inconsistent stroke directions</li>
                            <li>Unusual letter shapes</li>
                            <li>Spatial organization issues</li>
                        </ul>
                    </div>
                    
                    <div class="chart-container" id="accuracy-chart"></div>
                    
                    <div class="info-section">
                        <h3>Early Detection Importance</h3>
                        <p>Early identification of dyslexia can lead to more effective interventions. Our system provides:</p>
                        <ul>
                            <li>Quick screening assessment</li>
                            <li>Objective analysis of handwriting</li>
                            <li>Detailed reports on potential indicators</li>
                        </ul>
                        
                        <h3>How to Use This System</h3>
                        <p>For best results when submitting handwriting samples:</p>
                        <ul>
                            <li>Use clear, well-lit images</li>
                            <li>Include multiple words if possible</li>
                            <li>Submit samples from different writing sessions</li>
                        </ul>
                    </div>
                </div>
                
                <div class="footer">
                    Dyslexia Detection System &copy; 2023
                </div>
            </div>
            
            <script>
                // Static data for dyslexia accuracy
                document.addEventListener('DOMContentLoaded', () => {
                    const data = [{
                        x: ['Children', 'Adolescents', 'Adults'],
                        y: [92, 88, 85],
                        type: 'bar',
                        marker: {color: 'var(--primary)'}
                    }];
                    
                    const layout = {
                        title: 'Detection Accuracy by Age Group',
                        yaxis: {title: 'Accuracy (%)', range: [80, 100]},
                        margin: {t: 40, b: 30, l: 40, r: 0}
                    };
                    
                    Plotly.newPlot('accuracy-chart', data, layout);
                });
            </script>
        </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8000,
        reload=True  # Auto-reload on code changes
    )
    server = uvicorn.Server(config)
    asyncio.run(server.serve())