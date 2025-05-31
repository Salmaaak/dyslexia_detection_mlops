import pytest
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from src.models.model_predict import DyslexiaPredictor

# Integration Test - Uses actual small models
@pytest.fixture
def integration_predictor():
    return DyslexiaPredictor(
        feature_extractor_path="tests/test_assets/mini_feature_extractor.h5",
        classifier_path="tests/test_assets/mini_classifier.joblib"
    )

def test_integration_predict(integration_predictor):
    test_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8).tobytes()
    result = integration_predictor.predict(test_image)
    assert isinstance(result, dict)
    assert 'prediction' in result
    assert result['prediction'] in [0, 1]

# Unit Test - Uses dependency injection
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
    test_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8).tobytes()
    result = unit_predictor.predict(test_image)
    assert result['prediction'] == 0
    assert result['probability'] == 0.3

def test_error_handling():
    # Test error cases
    with pytest.raises(ValueError):
        DyslexiaPredictor()  # No arguments
        
    # Test invalid image
    predictor = DyslexiaPredictor(
        feature_extractor_path="tests/test_assets/mini_feature_extractor.h5",
        classifier_path="tests/test_assets/mini_classifier.joblib"
    )
    with pytest.raises(Exception):
        predictor.predict(b'invalid')