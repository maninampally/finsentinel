from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

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

    def dedupe_daily():
        from ingestion import dedup
        report = dedup.dedupe()
        print('dedupe report:', report)

    dedupe_task = PythonOperator(
        task_id='dedupe_bronze',
        python_callable=dedupe_daily,
    )

    def build_silver_daily():
        from ingestion import silver
        report = silver.build_silver()
        print('silver report:', report)

    silver_task = PythonOperator(
        task_id='build_silver',
        python_callable=build_silver_daily,
    )

    # dbt run: run staging model and tests
    dbt_task = BashOperator(
        task_id='dbt_run_staging',
        bash_command='cd dbt && dbt run -s stg_articles && dbt test -s stg_articles',
    )

    ingest_task >> dedupe_task >> silver_task >> dbt_task
