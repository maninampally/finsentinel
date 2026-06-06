from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    'owner':            'finsentinel',
    'retries':          1,
    'retry_delay':      timedelta(minutes=10),
    'email_on_failure': False,
}

with DAG(
    dag_id='finsentinel_retraining_dag',
    default_args=default_args,
    description='Drift-triggered FinBERT retraining pipeline',
    schedule_interval='0 6 * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['finsentinel', 'mlops', 'retraining']
) as dag:

    def check_drift(**context):
        from mlops.retrain_trigger import run_drift_check
        result = run_drift_check()
        context['ti'].xcom_push(key='drift_result', value=result)
        return 'retrain' if result['drift_detected'] else 'skip_retraining'

    def retrain():
        from ml.train_finbert import train_finbert
        train_finbert()

    def promote():
        from ml.evaluate import promote_best_model
        promote_best_model()

    drift_check = BranchPythonOperator(
        task_id='check_drift',
        python_callable=check_drift,
        provide_context=True
    )

    retrain_task = PythonOperator(
        task_id='retrain',
        python_callable=retrain
    )

    promote_task = PythonOperator(
        task_id='promote_best_model',
        python_callable=promote
    )

    skip = EmptyOperator(task_id='skip_retraining')

    drift_check >> [retrain_task, skip]
    retrain_task >> promote_task
