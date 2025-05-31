import pytest
from unittest.mock import patch, MagicMock
from src.models.model_predict import DyslexiaPredictor
import numpy as np

@pytest.fixture
def mock_predictor():
    with patch('tensorflow.keras.models.load_model') as mock_tf_load, \
         patch('joblib.load') as mock_joblib_load:
        # Mock the model loading functions to return dummy mocks
        mock_tf_load.return_value = MagicMock()
        mock_joblib_load.return_value = MagicMock()

        # Create DyslexiaPredictor instance with dummy paths
        predictor = DyslexiaPredictor(
            feature_extractor_path='dummy_path.h5',
            classifier_path='dummy_path.joblib'
        )
        yield predictor

def test_predict(mock_predictor):
    # Create dummy image bytes (128x128 uint8)
    mock_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8).tobytes()

    # Call predict on the mocked predictor instance
    result = mock_predictor.predict(mock_image)

    # Simple assertion to check we got a result (mocked)
    assert result is not None
