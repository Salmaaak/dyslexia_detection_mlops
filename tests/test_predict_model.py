import pytest
from unittest.mock import MagicMock, patch
import numpy as np
+ from src.models.model_predict import DyslexiaPredictor


@pytest.fixture
def mock_predictor():
    with patch('tensorflow.keras.models.load_model'), \
         patch('joblib.load'):
        predictor = DyslexiaPredictor(
            feature_extractor_path='dummy_path.h5',
            classifier_path='dummy_path.joblib'
        )
        
        # Mock the models
        predictor.feature_extractor.predict = MagicMock(
            return_value=np.random.rand(1, 256)  # Mock features
        )
        predictor.classifier.predict_proba = MagicMock(
            return_value=[[0.3, 0.7]]  # Mock probabilities
        )
        yield predictor

def test_predict(mock_predictor):
    # Create mock image bytes
    mock_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8).tobytes()
    
    result = mock_predictor.predict(mock_image)
    
    assert isinstance(result, dict)
    assert 'prediction' in result
    assert 'probability' in result
    assert 'class_probabilities' in result
    assert result['class_probabilities']['Normal'] + result['class_probabilities']['Reversal'] == pytest.approx(1.0)