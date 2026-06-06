import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def compare_models(experiment_name: str = "finsentinel-sentiment") -> pd.DataFrame:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_f1 DESC"]
    )

    records = []
    for run in runs:
        records.append({
            "run_name": run.data.tags.get("mlflow.runName"),
            "model":    run.data.params.get("model"),
            "val_f1":   run.data.metrics.get("val_f1"),
            "run_id":   run.info.run_id
        })

    df = pd.DataFrame(records)
    print(df.to_string(index=False))
    return df


def promote_best_model(model_name: str = "FinSentinel_Production"):
    client = MlflowClient()

    staging = client.get_latest_versions(model_name, stages=["Staging"])
    if not staging:
        print("No model in Staging.")
        return

    staging_f1 = float(client.get_metric_history(
        staging[0].run_id, "final_f1")[0].value)

    production = client.get_latest_versions(model_name, stages=["Production"])
    if production:
        prod_f1 = float(client.get_metric_history(
            production[0].run_id, "final_f1")[0].value)
        should_promote = staging_f1 > prod_f1
    else:
        should_promote = True

    if should_promote:
        client.transition_model_version_stage(
            name=model_name,
            version=staging[0].version,
            stage="Production",
            archive_existing_versions=True
        )
        print(f"Promoted v{staging[0].version} to Production (F1: {staging_f1:.4f})")
    else:
        print(f"Current Production is better. No promotion.")


if __name__ == "__main__":
    compare_models()
    promote_best_model()
