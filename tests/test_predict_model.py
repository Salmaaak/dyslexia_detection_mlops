import pytest
import numpy as np
import cv2
import tensorflow as tf
from sklearn.base import BaseEstimator
from src.models.model_predict import DyslexiaPredictor

def create_valid_test_image():
    """Create a valid grayscale test image that OpenCV can decode"""
    # Create a simple gradient image
    img = np.zeros((128, 128), dtype=np.uint8)
    for i in range(128):
        img[i, :] = i * 2  # Gradient from black to white
    
    # Encode as PNG bytes
    _, img_bytes = cv2.imencode('.png', img)
    return img_bytes.tobytes()

# Integration Test
@pytest.fixture
def integration_predictor():
    return DyslexiaPredictor(
        feature_extractor_path="tests/test_assets/mini_feature_extractor.h5",
        classifier_path="tests/test_assets/mini_classifier.joblib"
    )

def test_integration_predict(integration_predictor):
    test_image = create_valid_test_image()
    result = integration_predictor.predict(test_image)
    assert isinstance(result, dict)
    assert 'prediction' in result
    assert result['prediction'] in [0, 1]

# Unit Test
class MockClassifier(BaseEstimator):
    def predict(self, X):
        return np.array([0] * len(X))
    
    def predict_proba(self, X):
        return np.array([[0.7, 0.3]] * len(X))

@pytest.fixture
def unit_predictor():
    mock_fe = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(128, 128, 1))])
    mock_clf = MockClassifier()
    return DyslexiaPredictor(
        feature_extractor=mock_fe,
        classifier=mock_clf
    )

def test_unit_predict(unit_predictor):
    test_image = create_valid_test_image()
    result = unit_predictor.predict(test_image)
    assert result['prediction'] == 0
    assert result['probability'] == 0.3

def test_error_handling():
    # Test invalid initialization
    with pytest.raises(ValueError):
        DyslexiaPredictor()
        
    # Test invalid image
    predictor = DyslexiaPredictor(
        feature_extractor_path="tests/test_assets/mini_feature_extractor.h5",
        classifier_path="tests/test_assets/mini_classifier.joblib"
    )
    with pytest.raises(Exception):
        predictor.predict(b'invalid_image_data')