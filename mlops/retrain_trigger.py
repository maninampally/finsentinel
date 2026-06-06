import requests
from google.cloud import bigquery

import mlflow
from mlops.evidently_monitor import generate_monitoring_report, check_and_trigger_retraining


def load_reference_data(days: int = 30) -> object:
    client = bigquery.Client()
    query = f"""
        SELECT cleaned_text, sentiment AS prediction
        FROM `finsentinel-nlp.finsentinel_gold.predictions`
        WHERE published_at >= DATE_SUB(CURRENT_DATE(), INTERVAL {days * 2} DAY)
          AND published_at < DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
    """
    return client.query(query).to_dataframe()


def load_current_data(days: int = 30) -> object:
    client = bigquery.Client()
    query = f"""
        SELECT cleaned_text, sentiment AS prediction
        FROM `finsentinel-nlp.finsentinel_gold.predictions`
        WHERE published_at >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
    """
    return client.query(query).to_dataframe()


def run_drift_check():
    reference = load_reference_data()
    current   = load_current_data()
    result    = generate_monitoring_report(reference, current)
    check_and_trigger_retraining(result)
    return result


if __name__ == '__main__':
    run_drift_check()
