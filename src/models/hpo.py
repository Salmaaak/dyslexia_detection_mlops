# src/models/hpo.py

import optuna
import mlflow
import mlflow.sklearn
import os
from pathlib import Path
from train_model import train_random_forest, train_svm, load_features

def setup_mlflow():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # Hardcoded path
    mlflow.set_experiment("Dyslexia_Detection_HPO")

def objective(trial, model_type='rf'):
    X_train, X_val, _, y_train, y_val, _ = load_features()

    if model_type == 'rf':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'random_state': 42
        }
        model = train_random_forest(X_train, y_train, params)

    elif model_type == 'svm':
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'probability': True,
            'random_state': 42
        }
        model = train_svm(X_train, y_train, params)

    val_accuracy = model.score(X_val, y_val)
    return val_accuracy

def run_hpo(model_type='rf', n_trials=20):
    setup_mlflow()

    with mlflow.start_run(run_name=f"{model_type}_hpo"):
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, model_type), n_trials=n_trials)

        # Log best parameters and validation accuracy
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_accuracy", study.best_value)

        # Retrain on train+val and evaluate on test
        X_train, _, X_test, y_train, _, y_test = load_features()
        model = train_random_forest(X_train, y_train, study.best_params) if model_type == 'rf' \
                else train_svm(X_train, y_train, study.best_params)

        test_accuracy = model.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.sklearn.log_model(model, artifact_path="model")

    return study.best_params

if __name__ == '__main__':
    os.chdir(Path(__file__).parent.parent)

    print("Running HPO for Random Forest...")
    best_rf_params = run_hpo('rf', n_trials=5)

    print("\nRunning HPO for SVM...")
    best_svm_params = run_hpo('svm', n_trials=5)

    print("\nTo view results, run:")
    print(f"mlflow ui --backend-store-uri sqlite:///{Path.cwd() / 'mlflow.db'}")
