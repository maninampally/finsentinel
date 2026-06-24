import hashlib
import json
import os
import time
from typing import List, Optional
from urllib.parse import urljoin

import mlflow.pytorch
from transformers import BertTokenizer
import redis
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from databricks import sql

app = FastAPI(title="FinSentinel API", version="3.0")

# Config
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "https://adb-XXXXX.databricks.us")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_CATALOG = os.getenv("DATABRICKS_CATALOG", "finsentinel")
DATABRICKS_SCHEMA = os.getenv("DATABRICKS_SCHEMA", "gold")

# Redis cache
cache = redis.Redis(host='redis', port=6379, decode_responses=True)

# Model
try:
    model = mlflow.pytorch.load_model("models:/FinSentinel_Production/Production")
    model.eval()
except Exception as e:
    print(f"Warning: could not load model: {e}")
    model = None

# Tokenizer
try:
    tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
except Exception as e:
    tokenizer = None
    print(f"Warning: could not load tokenizer: {e}")

LABELS = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


class HeadlineRequest(BaseModel):
    text: str
    ticker: Optional[str] = None


class BatchRequest(BaseModel):
    headlines: List[str]
    ticker: Optional[str] = None


class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    scores: dict
    latency_ms: float


@app.post("/predict", response_model=SentimentResponse)
async def predict(request: HeadlineRequest):
    cache_key = f"pred:{hashlib.sha256(request.text.encode('utf-8')).hexdigest()}"
    cached = cache.get(cache_key)
    if cached:
        return SentimentResponse(**json.loads(cached))

    start = time.time()
    if tokenizer is None:
        raise HTTPException(status_code=500, detail="Tokenizer not loaded; ensure transformers models are available")

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_idx = probs.argmax().item()
    latency  = (time.time() - start) * 1000

    result = SentimentResponse(
        text=request.text,
        sentiment=LABELS[pred_idx],
        confidence=round(probs[pred_idx].item(), 4),
        scores={
            "negative": round(probs[0].item(), 4),
            "neutral":  round(probs[1].item(), 4),
            "positive": round(probs[2].item(), 4)
        },
        latency_ms=round(latency, 2)
    )

    cache.setex(cache_key, 3600, json.dumps(result.dict()))
    return result


@app.post("/batch_predict")
async def batch_predict(request: BatchRequest):
    return [
        await predict(HeadlineRequest(text=h, ticker=request.ticker))
        for h in request.headlines
    ]


@app.get("/sentiment/{ticker}")
async def ticker_sentiment(ticker: str, days: int = 7):
    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        raise HTTPException(status_code=500, detail="Databricks credentials not configured")

    try:
        with sql.connect(
            host=DATABRICKS_HOST.replace("https://", "").replace("http://", ""),
            token=DATABRICKS_TOKEN,
            http_path="/sql/1.0/warehouses/default"
        ) as connection:
            cursor = connection.cursor()
            query = f"""
                SELECT
                    date,
                    ticker,
                    article_count AS count,
                    source_count,
                    avg_word_count
                FROM {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.sentiment_features_gold
                WHERE ticker = %s
                  AND date >= CURRENT_DATE() - INTERVAL {days} DAY
                ORDER BY date DESC
            """
            cursor.execute(query, (ticker.upper(),))
            results = [dict(row) for row in cursor.fetchall()]
            return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Databricks query failed: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "FinBERT-finetuned"}


@app.get("/metrics")
async def metrics():
    return {
        "cache_hits":     cache.info().get("keyspace_hits", 0),
        "cache_misses":   cache.info().get("keyspace_misses", 0),
        "model_version":  "FinSentinel_Production/Production"
    }
