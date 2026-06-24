# Databricks notebook source
# MAGIC %md
# MAGIC # FinSentinel: Drift Monitoring & Retraining Trigger
# MAGIC Detect data drift and model drift, trigger retraining if threshold exceeded

# COMMAND ----------

import pandas as pd
from datetime import datetime, timedelta
import mlflow

# COMMAND ----------

# Config
DRIFT_THRESHOLD = float(dbutils.widgets.get("drift_threshold", "0.30"))
CATALOG = "finsentinel"
SCHEMA = "gold"

# COMMAND ----------

# Get baseline statistics (assume from 30 days ago)
baseline_df = spark.sql(f"""
SELECT
  COUNT(*) as total_articles,
  COUNT(DISTINCT ticker) as unique_tickers,
  AVG(word_count) as avg_word_count,
  COUNT(DISTINCT source) as unique_sources
FROM {CATALOG}.silver.articles_silver
WHERE published_at_ts >= DATE_SUB(CURRENT_DATE(), INTERVAL 60 DAY)
  AND published_at_ts < DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
""").toPandas()

# Get current statistics (last 7 days)
current_df = spark.sql(f"""
SELECT
  COUNT(*) as total_articles,
  COUNT(DISTINCT ticker) as unique_tickers,
  AVG(word_count) as avg_word_count,
  COUNT(DISTINCT source) as unique_sources
FROM {CATALOG}.silver.articles_silver
WHERE published_at_ts >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
""").toPandas()

# COMMAND ----------

# Calculate drift score (Euclidean distance on normalized metrics)
baseline = baseline_df.iloc[0]
current = current_df.iloc[0]

metrics = ['avg_word_count']
drift_deltas = []

for metric in metrics:
    baseline_val = baseline[metric]
    current_val = current[metric]

    if baseline_val > 0:
        delta = abs(current_val - baseline_val) / baseline_val
        drift_deltas.append(delta)

drift_score = sum(drift_deltas) / len(drift_deltas) if drift_deltas else 0

print(f"Drift Score: {drift_score:.2%}")
print(f"Threshold: {DRIFT_THRESHOLD:.2%}")

# COMMAND ----------

# Log metrics to MLflow
mlflow.start_run()
mlflow.log_metric("data_drift_score", drift_score)
mlflow.log_metric("drift_threshold", DRIFT_THRESHOLD)
mlflow.log_param("baseline_period", "60-30 days ago")
mlflow.log_param("current_period", "last 7 days")
mlflow.end_run()

# COMMAND ----------

# Decision: trigger retraining if drift > threshold
trigger_retraining = drift_score > DRIFT_THRESHOLD

print(f"\n{'TRIGGER RETRAINING' if trigger_retraining else 'SKIP RETRAINING'}")

# Store decision in Databricks secret for workflow coordination
dbutils.jobs.taskValues.set("should_retrain", str(trigger_retraining).lower())
