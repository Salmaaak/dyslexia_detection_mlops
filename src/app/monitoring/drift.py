import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_drift(current_features, reference_path="data/reference.csv"):
    """Check for data drift"""
    reference = pd.read_csv(reference_path)
    current = pd.DataFrame([current_features])
    
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    return report