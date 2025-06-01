import pandas as pd
import numpy as np

# Generate comprehensive synthetic reference data
np.random.seed(42)
data = {
    'feature_1': np.random.normal(0.2, 0.05, 1000),
    'feature_2': np.random.normal(0.45, 0.07, 1000),
    'feature_3': np.random.normal(0.1, 0.03, 1000),
    'feature_4': np.random.normal(0.85, 0.04, 1000),
    'feature_5': np.random.normal(0.3, 0.05, 1000),
    'feature_6': np.random.normal(0.75, 0.06, 1000),
    'feature_7': np.random.normal(0.05, 0.02, 1000),
    'feature_8': np.random.normal(0.9, 0.03, 1000),
    'predicted': np.random.randint(0, 2, 1000),
    'actual': np.random.randint(0, 2, 1000),
    'confidence': np.random.uniform(0.7, 1.0, 1000)
}

df = pd.DataFrame(data)
df.to_csv('data/reference.csv', index=False)