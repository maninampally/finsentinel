import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(tracking_uri: str = None, experiment_name: str = "finsentinel-sentiment"):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_model_to_registry(model, model_name: str, run_id: str = None):
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model" if run_id else f"models:/{model_name}/latest"
    mlflow.register_model(model_uri, model_name)


def get_production_model_uri(model_name: str = "FinSentinel_Production") -> str:
    return f"models:/{model_name}/Production"


def list_experiments():
    client = MlflowClient()
    for exp in client.search_experiments():
        print(f"{exp.experiment_id}: {exp.name}")
