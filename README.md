# FinSentinel

**Real-Time Financial News Sentiment Intelligence Platform**

Production-grade NLP + Data Engineering + MLOps on GCP — with an optional Databricks upgrade path.

Fine-tunes **ProsusAI/FinBERT** on FinancialPhraseBank to classify financial headlines as `POSITIVE / NEUTRAL / NEGATIVE` with ~0.95 F1. Ingests 500+ daily articles from NewsAPI, RSS, and Reddit, streams through GCP Pub/Sub → Dataflow → BigQuery, and serves predictions via FastAPI with a live Streamlit dashboard.

---

## Architecture

```
NewsAPI / RSS / Reddit / SEC EDGAR
          ↓
    GCP Pub/Sub  (real-time stream)
          ↓
  Dataflow — Apache Beam
  (clean · dedupe · ticker extract)
          ↓
  BigQuery  raw layer
          ↓
  dbt Core  staging → gold
          ↓
  FinBERT inference  (FastAPI · Redis cache)
          ↓
  Streamlit Dashboard  +  Evidently AI drift monitoring
          ↓
  GitHub Actions → Artifact Registry → GCE  (CI/CD)
```

**Databricks edition** (v2): replaces Dataflow + dbt with Structured Streaming + Delta Live Tables + Feature Store + Databricks Model Serving. See [PROJECT.md](PROJECT.md).

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
├── dataflow/
│   └── pipeline.py            # Apache Beam streaming pipeline (v1)
├── notebooks/                 # Databricks notebooks (v2)
│   ├── 01_bronze_ingestion.py
│   ├── 02_dlt_pipeline.py
│   ├── 03_feature_store.py
│   └── 04_train_finbert.py
├── dbt/                       # BigQuery transformations (v1)
│   └── models/
│       ├── staging/stg_articles.sql
│       └── gold/sentiment_features.sql
├── ml/
│   ├── train_finbert.py       # FinBERT fine-tuning (PyTorch)
│   ├── train_bilstm.py        # BiLSTM baseline (TensorFlow)
│   ├── train_lgbm.py          # LightGBM + TF-IDF baseline
│   └── evaluate.py            # Model comparison + registry promotion
├── mlops/
│   ├── evidently_monitor.py   # Drift detection
│   ├── mlflow_tracking.py     # MLflow helpers
│   └── retrain_trigger.py     # Auto retraining
├── api/
│   └── app.py                 # FastAPI — /predict, /batch_predict, /sentiment/{ticker}
├── dashboard/
│   └── streamlit_app.py       # Live sentiment dashboard
├── airflow/dags/
│   ├── ingestion_dag.py       # Scheduled ingestion (every 5 min)
│   └── retraining_dag.py      # Drift-triggered retraining
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/main.yml # CI/CD: test → build → push → deploy
```

---

## Quick Start

### 1. Clone + install

```bash
git clone https://github.com/<your-username>/finsentinel.git
cd finsentinel

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

> Apache Beam / Dataflow has a separate install to avoid dependency conflicts:
> ```bash
> pip install -r requirements-dataflow.txt
> ```

### 2. Environment variables

Copy and fill in `.env.example`:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GCP_PROJECT_ID` | Yes | GCP project ID |
| `NEWSAPI_KEY` | Recommended | NewsAPI.org key |
| `PUBSUB_TOPIC` | Yes (GCP) | Pub/Sub topic name |
| `MLFLOW_TRACKING_URI` | Yes (API) | MLflow server URI |
| `REDDIT_CLIENT_ID` | Optional | Reddit API client ID |
| `REDDIT_CLIENT_SECRET` | Optional | Reddit API secret |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes (GCP) | Path to GCP service account JSON |

### 3. Run ingestion locally (no GCP needed)

Writes articles to `data/bronze/YYYY-MM-DD.ndjson`. RSS works without any API keys.

```bash
python -m ingestion.runner
```

### 4. Run with Docker

```bash
docker compose up
```

Starts: FastAPI on `:8000` · Redis on `:6379` · Streamlit on `:8501`

---

## GCP Setup (v1: GCP-Native)

```bash
# Create project + enable APIs
gcloud projects create finsentinel-nlp
gcloud config set project finsentinel-nlp

gcloud services enable \
  pubsub.googleapis.com \
  dataflow.googleapis.com \
  bigquery.googleapis.com \
  artifactregistry.googleapis.com \
  compute.googleapis.com

# Pub/Sub
gcloud pubsub topics create financial-news-stream
gcloud pubsub subscriptions create news-processor \
  --topic=financial-news-stream

# BigQuery datasets
bq mk --dataset finsentinel_raw
bq mk --dataset finsentinel_gold
bq mk --dataset finsentinel_monitoring

# GCS bucket for Dataflow temp
gsutil mb -l us-central1 gs://finsentinel-artifacts
```

Then run the Pub/Sub publisher:

```bash
python -m ingestion.news_publisher
```

---

## Model Training

### FinBERT (primary)

```bash
python ml/train_finbert.py
```

Trains on [FinancialPhraseBank](https://huggingface.co/datasets/financial_phrasebank). Logs to MLflow. Registers best model as `FinSentinel_Production`.

### All 3 models + comparison

```bash
python ml/evaluate.py
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

- **MLflow** — experiment tracking across 20+ runs, model registry (Staging → Production)
- **Evidently AI** — text drift + prediction drift monitoring; auto-triggers Airflow retraining DAG when drift > 30%
- **GitHub Actions** — on push to `main`: run tests → build Docker image → push to Artifact Registry → deploy to GCE

---

## dbt (v1)

```bash
cd dbt
dbt deps
dbt run
dbt test
```

Builds `stg_articles` (staging) and `sentiment_features` (gold) tables in BigQuery.

---

## Databricks Edition (v2)

Full migration guide, notebook code, and Databricks-specific setup in [PROJECT.md](PROJECT.md#v2-databricks-edition--full-code).

Replaces: Dataflow → Structured Streaming, dbt → Delta Live Tables, DagsHub MLflow → Databricks Managed MLflow, FastAPI-only serving → Databricks Model Serving.

---

## Resume Bullets

**GCP-Native (v1)**
> Architected real-time financial news sentiment pipeline on GCP ingesting 500+ daily articles via Pub/Sub and Dataflow (Apache Beam) into BigQuery, with dbt Core transformations building ticker-level sentiment aggregations across a 4-tier medallion architecture.

**Databricks (v2)**
> Architected real-time financial news sentiment pipeline ingesting 500+ daily articles via GCP Pub/Sub into Databricks Structured Streaming, building a 3-tier Delta Lake medallion architecture (Bronze → Silver → Gold) with Delta Live Tables replacing dbt for declarative, pipeline-managed transformations.

Full bullets → [PROJECT.md § Resume Bullets](PROJECT.md#resume-bullets)

---

## Tech Stack

`PyTorch` · `HuggingFace Transformers` · `FinBERT` · `TensorFlow` · `LightGBM` · `MLflow` · `Evidently AI` · `Apache Beam` · `GCP Pub/Sub` · `Dataflow` · `BigQuery` · `dbt Core` · `FastAPI` · `Redis` · `Streamlit` · `Plotly` · `Docker` · `GitHub Actions` · `Databricks` · `Delta Lake` · `Airflow`