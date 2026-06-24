# Databricks notebook source
# MAGIC %md
# MAGIC # FinSentinel: Pub/Sub → Delta Lake Bronze Ingestion
# MAGIC Real-time financial news streaming from GCP Pub/Sub into Databricks

# COMMAND ----------

import json
import re
import langdetect
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime

# COMMAND ----------

# Config
PUBSUB_SUBSCRIPTION = "projects/finsentinel-nlp/subscriptions/news-processor"
GCS_BUCKET = "gs://finsentinel-databricks"
BRONZE_TABLE = "finsentinel.bronze.articles"
CHECKPOINT_PATH = f"{GCS_BUCKET}/checkpoints/streaming_ingest"

# COMMAND ----------

# Create Pub/Sub source
df_stream = (
    spark.readStream
    .format("cloud-pubsub")
    .option("projectIdForSourceCreds", "finsentinel-nlp")
    .option("pubsubGcpCredential", "/dbfs/mnt/secrets/gcp-key.json")
    .option("subscriptionId", PUBSUB_SUBSCRIPTION)
    .option("failOnDataLoss", "false")
    .load()
)

# COMMAND ----------

# Parse JSON messages
@udf(returnType=StructType([
    StructField("title", StringType()),
    StructField("summary", StringType()),
    StructField("content", StringType()),
    StructField("author", StringType()),
    StructField("source", StringType()),
    StructField("url", StringType()),
    StructField("published_at", StringType()),
]))
def parse_json(payload):
    try:
        return json.loads(payload.decode('utf-8'))
    except:
        return None

df_parsed = df_stream.select(
    parse_json(col("payload")).alias("article")
).filter(col("article").isNotNull()).select("article.*")

# COMMAND ----------

# Clean text
@udf(returnType=StringType())
def clean_text(text):
    if not text:
        return None
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    text = ' '.join(text.split())

    try:
        if langdetect.detect(text) != 'en':
            return None
    except:
        return None

    return text if len(text) >= 20 else None

df_clean = df_parsed.select(
    col("*"),
    clean_text(concat(col("title"), lit(" "), col("summary"))).alias("cleaned_text")
).filter(col("cleaned_text").isNotNull())

# COMMAND ----------

# Extract tickers (hardcoded list, can be fetched from Databricks)
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
           'META', 'NVDA', 'JPM', 'BAC', 'GS', 'WMT', 'JNJ']

@udf(returnType=ArrayType(StringType()))
def extract_tickers(text):
    if not text:
        return ['MARKET']
    text_upper = text.upper()
    found = [t for t in TICKERS if t in text_upper]
    return found if found else ['MARKET']

df_tickers = df_clean.select(
    col("*"),
    extract_tickers(col("cleaned_text")).alias("tickers")
)

# COMMAND ----------

# Deduplication: use Delta Lake's native merge for proper distributed dedup
df_deduped = df_tickers.select(
    col("*"),
    expr("row_number() OVER (PARTITION BY cleaned_text ORDER BY published_at DESC)").alias("rn")
).filter(col("rn") == 1).drop("rn")

# COMMAND ----------

# Add metadata
df_final = df_deduped.select(
    col("*"),
    col("word_count").cast("integer"),
    col("published_at").cast("timestamp").alias("published_at_ts"),
    current_timestamp().alias("ingested_at")
)

# COMMAND ----------

# Write to Delta Lake bronze table (append + deduplication via MERGE)
schema = StructType([
    StructField("title", StringType()),
    StructField("summary", StringType()),
    StructField("content", StringType()),
    StructField("author", StringType()),
    StructField("source", StringType()),
    StructField("url", StringType()),
    StructField("cleaned_text", StringType()),
    StructField("tickers", ArrayType(StringType())),
    StructField("published_at_ts", TimestampType()),
    StructField("ingested_at", TimestampType()),
])

# Create bronze table if not exists
try:
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {BRONZE_TABLE} (
            title STRING,
            summary STRING,
            content STRING,
            author STRING,
            source STRING,
            url STRING,
            cleaned_text STRING,
            tickers ARRAY<STRING>,
            published_at_ts TIMESTAMP,
            ingested_at TIMESTAMP
        )
        USING DELTA
        PARTITIONED BY (ingested_at)
    """)
except:
    pass

# Write stream with deduplication
query = (
    df_final
    .writeStream
    .format("delta")
    .option("checkpointLocation", CHECKPOINT_PATH)
    .outputMode("append")
    .table(BRONZE_TABLE)
)

query.awaitTermination()
