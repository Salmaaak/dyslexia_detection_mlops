import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from src.models.model_predict import DyslexiaPredictor
from fastapi import HTTPException

@pytest.fixture
def mock_predictor():
    with patch('tensorflow.keras.models.load_model') as mock_tf_load, \
         patch('joblib.load') as mock_joblib_load, \
         patch('builtins.open', mock_open()):
        
        # Setup mock models
        mock_fe = MagicMock()
        mock_fe.predict.return_value = np.random.rand(1, 256)
        
        mock_clf = MagicMock()
        mock_clf.predict.return_value = [1]
        mock_clf.predict_proba.return_value = [[0.3, 0.7]]
        
        mock_tf_load.return_value = mock_fe
        mock_joblib_load.return_value = mock_clf
        
        predictor = DyslexiaPredictor(
            feature_extractor_path='dummy_path.h5',
            classifier_path='dummy_path.joblib'
        )
        
        yield predictor, mock_fe, mock_clf

def test_predict_success(mock_predictor):
    predictor, mock_fe, mock_clf = mock_predictor
    test_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8).tobytes()
    
    result = predictor.predict(test_image)
    
    assert isinstance(result, dict)
    assert 'prediction' in result
    assert 'probability' in result
    assert 'class_probabilities' in result
    assert result['prediction'] in [0, 1]
    assert 0 <= result['probability'] <= 1

def test_predict_without_proba(mock_predictor):
    predictor, mock_fe, mock_clf = mock_predictor
    mock_clf.predict_proba = None  # Simulate classifier without predict_proba
    mock_clf.predict.return_value = [0.8]  # Continuous output
    
    test_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8).tobytes()
    result = predictor.predict(test_image)
    
    assert 0 <= result['probability'] <= 1

def test_model_loading_failure():
    with patch('tensorflow.keras.models.load_model', side_effect=Exception("Load error")), \
         patch('joblib.load', side_effect=Exception("Load error")), \
         pytest.raises(HTTPException):
        
        DyslexiaPredictor(
            feature_extractor_path='invalid.h5',
            classifier_path='invalid.joblib'
        )