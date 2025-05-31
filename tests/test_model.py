import pytest
from unittest.mock import MagicMock, patch
from src.models.model_predict import DyslexiaPredictor

@patch("src.models.model_predict.tf.keras.models.load_model")
@patch("src.models.model_predict.load")
def test_predict_with_mocked_models(mock_joblib_load, mock_keras_load):
    # Mock les modèles
    mock_feature_extractor = MagicMock()
    mock_feature_extractor.predict.return_value = [[0.1, 0.9]]  # Exemple de features

    mock_classifier = MagicMock()
    mock_classifier.predict.return_value = [1]
    mock_classifier.predict_proba.return_value = [[0.1, 0.9]]

    mock_keras_load.return_value = mock_feature_extractor
    mock_joblib_load.return_value = mock_classifier

    predictor = DyslexiaPredictor("dummy_path.h5", "dummy_model.pkl")

    # Fake image (128x128 grayscale)
    import numpy as np
    image = (np.random.rand(128, 128) * 255).astype(np.uint8)
    _, image_bytes = cv2.imencode('.png', image)
    image_bytes = image_bytes.tobytes()

    result = predictor.predict(image_bytes)

    assert result["prediction"] == 1
    assert 0.0 <= result["probability"] <= 1.0
    assert "Normal" in result["class_probabilities"]
    assert "Reversal" in result["class_probabilities"]
