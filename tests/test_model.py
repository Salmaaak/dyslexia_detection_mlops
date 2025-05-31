from src.models.predict_model import predict
from src.models.load_model import load_model
import numpy as np

def test_model_prediction():
    model = load_model()
    dummy_input = np.random.rand(1, *model.input_shape[1:])
    prediction = predict(model, dummy_input)
    assert prediction.shape[0] == 1
    print("Model prediction test passed!")