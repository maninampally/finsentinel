# FinSentinel

**Real-Time Financial News Sentiment Intelligence Platform**

Production-grade NLP + Data Engineering + MLOps on **Databricks**.

Fine-tunes **ProsusAI/FinBERT** on FinancialPhraseBank to classify financial headlines as `POSITIVE / NEUTRAL / NEGATIVE` with ~0.95 F1. Ingests 500+ daily articles from NewsAPI, RSS, and Reddit, streams through GCP Pub/Sub → Databricks Structured Streaming → Delta Lake, and serves predictions via FastAPI with a live Streamlit dashboard.

---

## Architecture

```
NewsAPI / RSS / Reddit / SEC EDGAR
          ↓
    GCP Pub/Sub  (real-time stream)
          ↓
  Databricks Structured Streaming
  (clean · dedupe · ticker extract)
          ↓
  Delta Lake  Bronze → Silver → Gold (medallion)
          ↓
  Delta Live Tables  transformations
          ↓
  FinBERT inference  (FastAPI · Redis cache)
          ↓
  Streamlit Dashboard  +  Databricks Model Monitoring
          ↓
  Databricks Workflows  (orchestration)
```

---

## Model Comparison

| Model | F1 | Notes |
|---|---|---|
| FinBERT (PyTorch) | ~0.95 | Production — industry standard |
| BiLSTM (TensorFlow) | ~0.86 | Benchmark |
| LightGBM + TF-IDF | ~0.74 | Baseline — fast, interpretable |

---

## Repo Structure

```
finsentinel/
├── ingestion/                 # News fetching + Pub/Sub publishing
│   ├── news_publisher.py      # Pub/Sub publisher (GCP)
│   ├── runner.py              # Local ingest → data/bronze/ (dev)
│   └── sources/
│       ├── newsapi_source.py
│       ├── rss_source.py
│       └── reddit_source.py
├── databricks/
│   ├── notebooks/
│   │   ├── 01_streaming_ingest.py      # Pub/Sub → Bronze (Structured Streaming)
│   │   ├── 02_drift_monitoring.py      # Drift detection + retraining trigger
│   │   └── 03_promote_model.py         # Model registry promotion
│   ├── dlt/
│   │   └── dlt_pipeline.sql            # Delta Live Tables (Bronze → Silver → Gold)
│   └── workflows/
│       └── finsentinel_workflow.yml     # Databricks Workflows orchestration
├── ml/
│   ├── train_finbert.py       # FinBERT fine-tuning (PyTorch + Databricks)
│   ├── train_bilstm.py        # BiLSTM baseline
│   └── train_lgbm.py          # LightGBM baseline
├── mlops/
│   ├── mlflow_tracking.py     # MLflow helpers
│   └── evidently_monitor.py   # Drift monitoring (optional)
├── api/
│   └── app.py                 # FastAPI — /predict, /batch_predict, /sentiment/{ticker}
├── dashboard/
│   └── streamlit_app.py       # Live sentiment dashboard (Databricks SQL)
├── tests/
│   └── test_api.py
├── scripts/
│   └── start_api_mock.py      # Local testing
├── Dockerfile
├── requirements.txt
├── MIGRATION_TO_DATABRICKS.md # Setup + migration guide
├── README.md
└── .github/workflows/main.yml # CI/CD (GitHub Actions)
```

---

## Quick Start

### 1. Prerequisites

- **Databricks workspace** (AWS/Azure/GCP)
- **GCP Pub/Sub** topic: `financial-news-stream`
- **GCS bucket**: `gs://finsentinel-databricks/`

### 2. Clone + install

```bash
git clone https://github.com/<your-username>/finsentinel.git
cd finsentinel

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Environment variables

Copy and update `.env`:

```bash
# Databricks
DATABRICKS_HOST=https://adb-XXXXXXX.databricks.us
DATABRICKS_TOKEN=your_token
DATABRICKS_CATALOG=finsentinel
DATABRICKS_SCHEMA=gold

# NewsAPI
NEWSAPI_KEY=your_key

# Redis
REDIS_HOST=redis
REDIS_PORT=6379

# MLflow
MLFLOW_TRACKING_URI=databricks
```

### 4. Setup Databricks

See [MIGRATION_TO_DATABRICKS.md](MIGRATION_TO_DATABRICKS.md) for step-by-step setup.

Quick steps:
```bash
# Create cluster, mount GCS, create schemas
# Deploy notebooks to /Workspace/finsentinel
# Create Delta Live Tables pipeline
# Create Databricks Workflow (every 5 minutes)
```

### 5. Run API locally (dev)

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple beats earnings"}'
```

### 6. Run dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Visit: `http://localhost:8501`

---

## GCP Setup (Pub/Sub + Ingestion)

```bash
# Create GCP project
gcloud projects create finsentinel-nlp
gcloud config set project finsentinel-nlp

# Enable Pub/Sub
gcloud services enable pubsub.googleapis.com

# Create Pub/Sub topic
gcloud pubsub topics create financial-news-stream
gcloud pubsub subscriptions create news-processor \
  --topic=financial-news-stream

# Create GCS bucket
gsutil mb -l us-central1 gs://finsentinel-databricks/

# Grant Databricks service account permissions
# (done in Databricks workspace secrets setup)
```

Pub/Sub ingestion:
```bash
python -m ingestion.news_publisher
```

---

## Model Training

### FinBERT (Databricks Job)

Training runs automatically via Databricks Workflow every 5 minutes (if drift detected).

Manual training on Databricks:
```python
# In Databricks notebook
%run /Workspace/finsentinel/04_train_finbert

# Trains on articles from finsentinel.silver.articles_silver
# Logs to Databricks MLflow
# Registers as FinSentinel_Production
```

Local training (for development):
```bash
python ml/train_finbert.py
```

### All 3 models (baseline comparison)

```bash
python ml/evaluate.py  # Compares FinBERT vs BiLSTM vs LightGBM
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Single headline → sentiment + confidence scores |
| `/batch_predict` | POST | CSV of headlines → bulk predictions |
| `/sentiment/{ticker}` | GET | Aggregated sentiment for ticker over N days |
| `/health` | GET | Health check |
| `/metrics` | GET | Cache hit rate + model version |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple reports record quarterly earnings, beats estimates"}'
```

```json
{
  "text": "Apple reports record quarterly earnings, beats estimates",
  "sentiment": "POSITIVE",
  "confidence": 0.9821,
  "scores": {"negative": 0.0041, "neutral": 0.0138, "positive": 0.9821},
  "latency_ms": 34.7
}
```

---

## MLOps

- **Databricks MLflow** — experiment tracking, model registry (Staging → Production) managed natively
- **Drift Monitoring** — custom Databricks notebook detects data drift; auto-triggers retraining when drift > 30%
- **Databricks Workflows** — orchestrates streaming, DLT, drift checks, and retraining every 5 minutes
- **Model Registry** — centralized model versioning + stage transitions (Staging → Production → Archived)

---

## Resume Bullets

> Architected real-time financial news sentiment pipeline ingesting 500+ daily articles via GCP Pub/Sub into Databricks Structured Streaming, building 3-tier Delta Lake medallion (Bronze → Silver → Gold) with Delta Live Tables transformations and Databricks Workflows orchestration. Integrated FinBERT fine-tuning (PyTorch) with automatic drift detection triggering retraining via MLflow Model Registry. Deployed FastAPI inference service with Redis caching and Streamlit dashboard connected to Databricks SQL endpoints.

---

## Tech Stack

**ML:** `PyTorch` · `HuggingFace Transformers` · `FinBERT` · `TensorFlow` · `LightGBM` · `scikit-learn`

**Data:** `Databricks` · `Delta Lake` · `Apache Spark` · `Structured Streaming` · `Delta Live Tables` · `GCP Pub/Sub`

**MLOps:** `Databricks MLflow` · `Databricks Workflows` · `Databricks Model Registry`

**API & Dashboard:** `FastAPI` · `Streamlit` · `Redis` · `Plotly` · `Databricks SQL`

**DevOps:** `Docker` · `GitHub Actions` · `GCP`