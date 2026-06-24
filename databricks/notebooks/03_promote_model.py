# Databricks notebook source
# MAGIC %md
# MAGIC # FinSentinel: Model Registry Promotion
# MAGIC Promote best-performing model from Staging → Production

# COMMAND ----------

import mlflow
import mlflow.pytorch
from mlflow.entities.metric_threshold import MetricThreshold

# COMMAND ----------

# Config
MODEL_NAME = "FinSentinel_Production"
REGISTRY_URI = "databricks-uc"
MIN_F1_THRESHOLD = 0.92

# COMMAND ----------

# Get latest Staging version
client = mlflow.tracking.MlflowClient(tracking_uri="databricks")

# Query model versions in Staging
try:
    staging_versions = client.get_model_version_by_stage(MODEL_NAME, "Staging")
    if staging_versions:
        latest_staging = staging_versions[-1]
        print(f"Latest Staging version: {latest_staging.version}")
        print(f"Status: {latest_staging.status}")

        # Check F1 score
        run = mlflow.get_run(latest_staging.run_id)
        f1_score = run.data.metrics.get("final_f1", 0)

        print(f"F1 Score: {f1_score:.4f}")

        if f1_score >= MIN_F1_THRESHOLD:
            # Promote to Production
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest_staging.version,
                stage="Production",
                archive_existing_versions=True
            )
            print(f"✅ Promoted version {latest_staging.version} to Production")
        else:
            print(f"❌ F1 score {f1_score:.4f} below threshold {MIN_F1_THRESHOLD}")
    else:
        print("❌ No Staging versions found")
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

# List all versions
all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
for version in all_versions:
    print(f"Version {version.version}: {version.current_stage}")
