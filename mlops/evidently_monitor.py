import pandas as pd
import requests
from evidently.metric_preset import ClassificationPreset, DataDriftPreset, TextOverviewPreset
from evidently.metrics import ColumnDriftMetric, DatasetMissingValuesSummary
from evidently.report import Report


def generate_monitoring_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame
) -> dict:
    report = Report(metrics=[
        ClassificationPreset(),
        DataDriftPreset(),
        TextOverviewPreset(column_name="cleaned_text"),
        ColumnDriftMetric(column_name="prediction"),
        DatasetMissingValuesSummary()
    ])

    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("monitoring/report.html")
    result = report.as_dict()

    drift_detected = result['metrics'][1]['result']['dataset_drift']
    drift_share    = result['metrics'][1]['result']['drift_share']

    return {
        "drift_detected": drift_detected,
        "drift_share":    drift_share,
        "report_path":    "monitoring/report.html"
    }


def check_and_trigger_retraining(drift_result: dict):
    if drift_result["drift_share"] > 0.3:
        print(f"DRIFT ALERT: {drift_result['drift_share']:.2%} features drifted — triggering retraining")
        trigger_airflow_dag("finsentinel_retraining_dag")
    else:
        print(f"No significant drift: {drift_result['drift_share']:.2%}")


def trigger_airflow_dag(dag_id: str):
    requests.post(
        f"http://airflow:8080/api/v1/dags/{dag_id}/dagRuns",
        json={"conf": {"triggered_by": "drift_monitor"}},
        auth=("admin", "admin")
    )
