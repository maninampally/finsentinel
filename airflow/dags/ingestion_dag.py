from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    'owner':            'finsentinel',
    'retries':          2,
    'retry_delay':      timedelta(minutes=5),
    'email_on_failure': False,
}

with DAG(
    dag_id='finsentinel_ingestion',
    default_args=default_args,
    description='Fetch financial news and publish to Pub/Sub every 5 minutes',
    schedule_interval='*/5 * * * *',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['finsentinel', 'ingestion']
) as dag:

    def publish_news():
        from ingestion.runner import ingest_once
        count = ingest_once()
        print(f"Ingested {count} articles")

    ingest_task = PythonOperator(
        task_id='publish_news_to_pubsub',
        python_callable=publish_news,
    )
