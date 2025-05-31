import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.models.model_predict import DyslexiaPredictor

# Patch decorators apply the patches BEFORE the fixture function runs, so __init__ uses the mocks
@pytest.fixture
@patch('src.models.model_predict.tf.keras.models.load_model')
@patch('src.models.model_predict.load')  # patch joblib.load used in your code
def mock_predictor(mock_joblib_load, mock_tf_load):
    # Setup the mocks to return MagicMock model objects
    mock_tf_load.return_value = MagicMock()
    mock_joblib_load.return_value = MagicMock()

    predictor = DyslexiaPredictor(
        feature_extractor_path='dummy_path.h5',
        classifier_path='dummy_path.joblib'
    )

    # Mock the predict methods to avoid real calls
    predictor.feature_extractor.predict = MagicMock(
        return_value=np.random.rand(1, 256)  # Mock features vector shape (adjust as needed)
    )
    predictor.classifier.predict_proba = MagicMock(
        return_value=[[0.3, 0.7]]  # Mock probabilities summing to 1
    )
    predictor.classifier.predict = MagicMock(return_value=[1])

    yield predictor

def test_predict(mock_predictor):
    import numpy as np
    # Create mock image bytes
    mock_image = np.random.randint(0, 255, (128, 128), dtype=np.uint8).tobytes()

    result = mock_predictor.predict(mock_image)

    assert isinstance(result, dict)
    assert 'prediction' in result
    assert 'probability' in result
    assert 'class_probabilities' in result

    # Probabilities sum approximately to 1
    total_prob = sum(result['class_probabilities'].values())
    assert total_prob == pytest.approx(1.0)
