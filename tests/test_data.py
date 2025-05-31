import numpy as np
from src.data.make_dataset import load_data

def test_data_shapes():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    assert X_train.shape[0] == y_train.shape[0]
    assert X_val.shape[0] == y_val.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    print("Data shape tests passed!")