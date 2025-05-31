import numpy as np
from src.data.make_dataset import load_data

from unittest.mock import patch

@patch('src.data.make_dataset.load_data')
def test_data_shapes(mock_load_data):
    import numpy as np
    # Créer des données factices compatibles
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 2, 10)
    X_val = np.random.rand(5, 5)
    y_val = np.random.randint(0, 2, 5)
    X_test = np.random.rand(3, 5)
    y_test = np.random.randint(0, 2, 3)

    mock_load_data.return_value = (X_train, y_train, X_val, y_val, X_test, y_test)

    X_train_out, y_train_out, X_val_out, y_val_out, X_test_out, y_test_out = load_data()

    assert X_train_out.shape[0] == y_train_out.shape[0]
    assert X_val_out.shape[0] == y_val_out.shape[0]
    assert X_test_out.shape[0] == y_test_out.shape[0]

    print("Data shape tests passed!")
