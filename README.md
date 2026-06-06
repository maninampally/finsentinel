# FinSentinel

## Setup

Install the core project dependencies first:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Install the optional Apache Beam / Dataflow stack only when you need `dataflow/pipeline.py`:

```powershell
python -m pip install -r requirements-dataflow.txt
```

## Ingestion

Run the local ingestion job to write raw articles to `data/bronze/`:

```powershell
python -m ingestion.runner
```

Required environment variables are optional for RSS-only runs. If present, `NEWSAPI_KEY`, `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT` enable the extra sources.