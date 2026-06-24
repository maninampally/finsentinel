# FinSentinel: Setup Guide

**Status:** Databricks-based architecture

## Previous Architecture (Removed)
```
Pub/Sub → Dataflow (Beam) → BigQuery → dbt → FastAPI
```

## Current Architecture
```
Pub/Sub → Databricks Structured Streaming → Delta Lake → DLT → FastAPI + Databricks SQL
```

---

## Setup Instructions

### Prerequisites
- Databricks workspace (AWS/Azure/GCP)
- GCP Pub/Sub topic: `financial-news-stream`
- GCS bucket: `gs://finsentinel-databricks/`
- Personal access token or service principal

### 1. Create Databricks Workspace

```bash
# Go to https://databricks.com
# Create workspace in your cloud (AWS/Azure/GCP)
# Get workspace URL: https://adb-XXXXXXX.databricks.us
```

### 2. Create Cluster

**Admin Console → Compute → Create Cluster**

```yaml
Name: finsentinel-cluster
Runtime: 15.3 LTS (Spark 3.5)
Worker type: i3.xlarge
Min workers: 1
Max workers: 4
Auto-scaling: enabled
```

### 3. Configure GCP Access

**Workspace Settings → Secrets → Create Secret Scope**

```bash
databricks secrets create-scope --scope finsentinel-gcp

# Store GCP service account key
databricks secrets put-secret --scope finsentinel-gcp --key sa-key --string-value @path/to/sa-key.json
```

### 4. Mount GCS Bucket

**In Databricks notebook:**

```python
dbutils.fs.mount(
  source = "gs://finsentinel-databricks",
  mount_point = "/mnt/finsentinel-gcs",
  extra_configs = {
    "google_service_account_key_path": "/dbfs/mnt/secrets/gcp-sa-key.json"
  }
)
```

### 5. Create Catalogs & Schemas

```sql
CREATE CATALOG IF NOT EXISTS finsentinel;
CREATE SCHEMA IF NOT EXISTS finsentinel.bronze;
CREATE SCHEMA IF NOT EXISTS finsentinel.silver;
CREATE SCHEMA IF NOT EXISTS finsentinel.gold;
```

### 6. Deploy Notebooks

Upload `databricks/notebooks/` to Databricks workspace:

```bash
databricks workspace mkdirs /Workspace/finsentinel
databricks workspace import -o ./databricks/notebooks/01_streaming_ingest.py /Workspace/finsentinel/01_streaming_ingest
databricks workspace import -o ./databricks/notebooks/02_drift_monitoring.py /Workspace/finsentinel/02_drift_monitoring
databricks workspace import -o ./databricks/notebooks/03_promote_model.py /Workspace/finsentinel/03_promote_model
```

### 7. Create Delta Live Tables Pipeline

**Workflows → Create Delta Live Table Pipeline**

```
Pipeline name: finsentinel-dlt
Notebook: /Workspace/finsentinel/dlt_pipeline.sql
Cluster: finsentinel-cluster
```

Or use CLI:

```bash
databricks pipelines create --config databricks/pipelines/dlt_config.json
```

### 8. Create Workflow

**Workflows → Create Job**

Upload YAML:

```bash
databricks jobs create --json-file databricks/workflows/finsentinel_workflow.yml
```

Or manually create with schedule: **Every 5 minutes**

### 9. Environment Variables

Create `.env`:

```bash
# Databricks
DATABRICKS_HOST=https://adb-XXXXXXX.databricks.us
DATABRICKS_TOKEN=your_token_here
DATABRICKS_CATALOG=finsentinel
DATABRICKS_SCHEMA=gold

# Redis (for API caching)
REDIS_HOST=redis
REDIS_PORT=6379

# MLflow
MLFLOW_TRACKING_URI=databricks
```

### 10. Update API & Dashboard

```bash
# Install new dependencies
pip install -r requirements.txt

# Run FastAPI
uvicorn api.app:app --host 0.0.0.0 --port 8000

# Run Streamlit
streamlit run dashboard/streamlit_app.py
```

---

## Data Migration

### Migrate Historical Data (Optional)

If you have historical data in BigQuery:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("finsentinel-migrate").getOrCreate()

# Read from BigQuery
df = spark.read.format("bigquery").option("table", "finsentinel-nlp.finsentinel_raw.articles").load()

# Write to Delta Lake bronze
df.write.mode("overwrite").save("gs://finsentinel-databricks/bronze/historical")
```

---

## Component Mapping: Previous → Current

| Component | Previous | Current |
|-----------|----|----|
| **Ingestion** | Dataflow (Apache Beam) | Databricks Structured Streaming |
| **Transform** | dbt Core SQL | Delta Live Tables SQL |
| **Storage** | BigQuery | Delta Lake (Databricks) |
| **Feature Store** | Manual | Databricks Feature Store (optional) |
| **Model Training** | Local PyTorch + MLflow | Databricks Jobs + Databricks MLflow |
| **Model Registry** | MLflow (external) | Databricks MLflow (built-in) |
| **Model Serving** | FastAPI | FastAPI (kept) or Databricks Model Serving |
| **Orchestration** | Apache Airflow | Databricks Workflows |
| **Monitoring** | Evidently AI | Databricks Model Monitoring + custom drift |
| **API** | FastAPI on VM | FastAPI on VM (same) |
| **Dashboard** | Streamlit + BigQuery | Streamlit + Databricks SQL |

---

## Removed Directories

- `dataflow/` — Beam pipeline (replaced by Structured Streaming)
- `dbt/` — dbt Core models (replaced by Delta Live Tables)
- `airflow/` — Airflow DAGs (replaced by Databricks Workflows)
- `Dockerfile.dbt` — dbt container
- `docker-compose.yml` — v1 orchestration

---

## Testing

### Test Streaming Pipeline

```python
# In Databricks notebook
display(spark.sql("SELECT * FROM finsentinel.bronze.articles LIMIT 10"))
```

### Test DLT

```python
# Check silver table
display(spark.sql("SELECT * FROM finsentinel.silver.articles_silver LIMIT 10"))

# Check gold table
display(spark.sql("SELECT * FROM finsentinel.gold.sentiment_features_gold LIMIT 10"))
```

### Test API Endpoint

```bash
curl -X GET "http://localhost:8000/sentiment/AAPL?days=7"
```

### Test Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

---

## Troubleshooting

### Pub/Sub Connection Failed

- Verify GCP service account has `pubsub.subscriber` role
- Check subscription exists: `gcloud pubsub subscriptions list`
- Test locally: `python -m ingestion.newsapi_source`

### Databricks SQL Endpoint Timeout

- Ensure cluster is running
- Check SQL endpoint: `Settings → SQL Endpoints`
- Verify firewall rules allow connection

### Model Loading Failed

- Check MLflow tracking URI: `MLFLOW_TRACKING_URI=databricks`
- Verify model exists in registry: `Model Registry → FinSentinel_Production`
- Train initial model: `python ml/train_finbert.py`

### DLT Pipeline Errors

- Check notebook syntax
- Verify schema exists: `SHOW SCHEMAS IN finsentinel`
- Review Databricks job logs

---

## Performance Tuning

### Streaming Checkpoint

```python
# Optimize checkpoint location
spark.conf.set("spark.sql.streaming.checkpointLocation", 
               "gs://finsentinel-databricks/checkpoints/streaming")
```

### Delta Lake Optimization

```sql
-- Compact small files (run weekly)
OPTIMIZE finsentinel.silver.articles_silver;
OPTIMIZE finsentinel.gold.sentiment_features_gold;

-- Check table stats
ANALYZE TABLE finsentinel.silver.articles_silver COMPUTE STATISTICS;
```

### Cluster Right-Sizing

For streaming:
- **i3.xlarge** (2 workers) handles ~500 articles/day
- **i3.2xlarge** (4 workers) handles ~5000 articles/day

For training:
- **g4dn.xlarge** (GPU) for PyTorch fine-tuning
- Auto-scale 1-4 workers

---

## Next Steps

1. ✅ Clean up v1 code (done)
2. ✅ Create Databricks notebooks (done)
3. ⏳ Set up Databricks workspace (you do this)
4. ⏳ Deploy notebooks + workflows
5. ⏳ Test streaming pipeline
6. ⏳ Verify data in Delta Lake
7. ⏳ Run model training on Databricks
8. ⏳ Deploy API + dashboard

---

## Support

For issues, check:
- [Databricks Documentation](https://docs.databricks.com)
- [Delta Lake Guide](https://docs.delta.io/)
- [Databricks MLflow](https://docs.databricks.com/applications/mlflow/)
