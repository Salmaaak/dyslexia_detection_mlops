# tests/create_test_models.py
import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from joblib import dump

# Create tiny feature extractor
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128, 1))
])
model.save("tests/test_assets/mini_feature_extractor.h5")

# Create tiny classifier
clf = LogisticRegression()
X = np.random.rand(10, 128*128)  # Match feature extractor output shape
y = np.random.randint(0, 2, 10)
clf.fit(X, y)
dump(clf, "tests/test_assets/mini_classifier.joblib")