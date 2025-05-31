import shap
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from pathlib import Path
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def load_best_model_from_mlflow(experiment_name='Dyslexia_Detection_HPO'):
    """Load the best model from MLflow experiments"""
    mlflow.set_tracking_uri("sqlite:///" + str(Path(__file__).parent.parent.parent / "mlflow.db"))
    client = MlflowClient()
    
    # Get the best run
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["metrics.test_accuracy DESC"],
        max_results=1
    )
    best_run = runs[0]
    model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
    return model, best_run.data.params

def load_and_scale_features():
    """Load and scale features with memory efficiency"""
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'features'
    X_train = np.load(data_dir / 'X_train.npy')
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    # Scale features in batches if needed
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_test

def calculate_shap_values_svm(model, X_train, X_test, nsamples=50):
    """Memory-efficient SHAP for SVM"""
    # Use a subset of the training data for background
    background = shap.utils.sample(X_train, nsamples)
    
    # Use LinearExplainer if kernel='linear', else use approximate methods
    if model.kernel == 'linear':
        explainer = shap.LinearExplainer(model, background)
    else:
        explainer = shap.KernelExplainer(model.predict, background)
    
    # Calculate SHAP values for a subset of test samples
    test_samples = X_test[:100]  # Adjust based on your memory
    shap_values = explainer.shap_values(test_samples)
    return explainer, shap_values, test_samples

def visualize_and_log_shap(model, shap_values, samples, params):
    """Generate and save SHAP visualizations"""
    reports_dir = Path(__file__).parent.parent.parent / 'reports' / 'shap'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, samples, show=False)
    plt.title(f"SVM SHAP Summary (C={params['C']}, kernel={params['kernel']})")
    plt.savefig(reports_dir / 'shap_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Log to MLflow
    mlflow.start_run()
    mlflow.log_artifact(str(reports_dir / 'shap_summary.png'))
    mlflow.end_run()

if __name__ == '__main__':
    try:
        print("1. Loading best SVM model from MLflow...")
        model, params = load_best_model_from_mlflow()
        
        print("2. Loading and scaling features...")
        X_train, X_test, y_test = load_and_scale_features()
        
        print("3. Calculating SHAP values (this may take a while)...")
        explainer, shap_values, samples = calculate_shap_values_svm(
            model, 
            X_train, 
            X_test,
            nsamples=10  # Reduce this if memory constrained
        )
        
        print("4. Generating visualizations...")
        visualize_and_log_shap(model, shap_values, samples, params)
        
        print("SHAP analysis completed successfully!")
    except Exception as e:
        print(f"Failed: {str(e)}")
        print("Try reducing nsamples or using fewer test samples")