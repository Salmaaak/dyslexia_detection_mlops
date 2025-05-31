# src/models/register_model.py

import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import joblib  # Ajouté pour l'enregistrement local
import os  # Ajouté pour la gestion des dossiers

def register_best_model(experiment_name='Dyslexia_Detection_HPO'):
    # Use the same absolute path as hpo.py
    db_path = Path("C:/Users/dell/Desktop/dyslexia/src/mlflow.db").absolute()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    print(f"Using MLflow database at: {mlflow.get_tracking_uri()}")
    
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Debugging: List all experiments
    print("\nAvailable experiments:")
    experiments = client.search_experiments()
    for exp in experiments:
        print(f"- {exp.name} (ID: {exp.experiment_id})")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    print(f"\nSearching runs in experiment '{experiment_name}' (ID: {experiment.experiment_id})")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_accuracy DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found in the experiment. Did you run hpo.py first?")
    
    best_run = runs[0]
    print(f"\nFound best run: {best_run.info.run_id}")
    print(f"Metrics: {best_run.data.metrics}")
    
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "dyslexia-detector"

    # Charger le modèle depuis MLflow
    model = mlflow.sklearn.load_model(model_uri)
    
    # Enregistrement local du modèle - NOUVEAU CODE AJOUTÉ
    local_model_dir = Path("C:/Users/dell/Desktop/dyslexia/models")
    local_model_dir.mkdir(exist_ok=True)  # Crée le dossier si inexistant
    local_model_path = local_model_dir / "best_model.joblib"
    joblib.dump(model, local_model_path)
    print(f"\nModel saved locally at: {local_model_path}")

    try:
        client.create_registered_model(model_name)
        print(f"Created new registered model '{model_name}'")
    except:
        print(f"Model '{model_name}' already exists")

    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run.info.run_id
    )
    print(f"Created model version {mv.version}")

    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production"
    )
    print(f"Transitioned version {mv.version} to Production")

    print(f"\nSuccessfully registered model '{model_name}' with test_accuracy={best_run.data.metrics['test_accuracy']:.4f}")

if __name__ == '__main__':
    register_best_model()