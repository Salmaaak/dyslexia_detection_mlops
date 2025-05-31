from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab
from evidently.model_profile import Profile
from evidently.profile_sections import DataDriftProfileSection
import numpy as np

def validate_data(X_train, X_test):
    # Convert to DataFrame for Evidently
    # (Mock conversion - adjust based on your actual data structure)
    train_data = {'feature': X_train.reshape(X_train.shape[0], -1).mean(axis=1)}
    test_data = {'feature': X_test.reshape(X_test.shape[0], -1).mean(axis=1)}
    
    data_drift_dashboard = Dashboard(tabs=[DataDriftTab])
    data_drift_dashboard.calculate(train_data, test_data)
    data_drift_dashboard.save('reports/data_drift.html')
    
    return data_drift_dashboard