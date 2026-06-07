import time
from typing import List, Optional

import mlflow.pytorch
from transformers import BertTokenizer
import redis
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="FinSentinel API", version="2.0")

cache = redis.Redis(host='redis', port=6379, decode_responses=True)
model = mlflow.pytorch.load_model("models:/FinSentinel_Production/Production")
model.eval()

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
    cache_key = f"pred:{hash(request.text)}"
    cached = cache.get(cache_key)
    if cached:
        return eval(cached)

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

    cache.setex(cache_key, 3600, str(result.dict()))
    return result


@app.post("/batch_predict")
async def batch_predict(request: BatchRequest):
    return [
        await predict(HeadlineRequest(text=h, ticker=request.ticker))
        for h in request.headlines
    ]


@app.get("/sentiment/{ticker}")
async def ticker_sentiment(ticker: str, days: int = 7):
    from google.cloud import bigquery
    client = bigquery.Client()
    query = f"""
        SELECT
            DATE(published_at)   AS date,
            sentiment,
            COUNT(*)             AS count,
            AVG(confidence)      AS avg_confidence
        FROM `finsentinel-nlp.finsentinel_gold.predictions`
        WHERE ticker = '{ticker.upper()}'
          AND published_at >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
        GROUP BY date, sentiment
        ORDER BY date DESC
    """
    results = client.query(query).to_dataframe()
    return results.to_dict(orient='records')


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
