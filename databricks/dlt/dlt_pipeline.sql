-- Databricks notebook source
-- Delta Live Tables Pipeline: Bronze → Silver → Gold
-- MAGIC %md
-- MAGIC # FinSentinel DLT Pipeline
-- MAGIC Medallion Architecture: Bronze (raw) → Silver (clean) → Gold (aggregated)

-- COMMAND ----------

CREATE OR REPLACE TABLE LIVE.articles_bronze
COMMENT "Raw articles from Pub/Sub streaming"
USING DELTA
PARTITIONED BY (ingested_at)
AS
SELECT
  title,
  summary,
  content,
  author,
  source,
  url,
  cleaned_text,
  tickers,
  published_at_ts,
  ingested_at
FROM finsentinel.bronze.articles;

-- COMMAND ----------

CREATE OR REPLACE TABLE LIVE.articles_silver
COMMENT "Cleaned, validated articles ready for ML"
TBLPROPERTIES (
  "quality" = "gold",
  "retention_days" = "90"
)
AS
SELECT
  md5(url) AS article_id,
  title,
  cleaned_text,
  content,
  author,
  source,
  url,
  explode(tickers) AS ticker,
  published_at_ts,
  ingested_at,
  length(cleaned_text) AS text_length,
  size(split(cleaned_text, '\s+')) AS word_count
FROM LIVE.articles_bronze
WHERE
  cleaned_text IS NOT NULL
  AND length(cleaned_text) >= 20
  AND published_at_ts IS NOT NULL
QUALIFY row_number() OVER (PARTITION BY md5(url) ORDER BY ingested_at DESC) = 1;

-- COMMAND ----------

CREATE OR REPLACE TABLE LIVE.sentiment_features_gold
COMMENT "Aggregated sentiment metrics by ticker and date"
TBLPROPERTIES (
  "quality" = "gold",
  "retention_days" = "365"
)
AS
SELECT
  DATE(published_at_ts) AS date,
  ticker,
  COUNT(*) AS article_count,
  COUNT(DISTINCT source) AS source_count,
  AVG(word_count) AS avg_word_count,
  MIN(published_at_ts) AS min_published_at,
  MAX(published_at_ts) AS max_published_at,
  current_timestamp() AS updated_at
FROM LIVE.articles_silver
GROUP BY
  DATE(published_at_ts),
  ticker;

-- COMMAND ----------

CREATE OR REPLACE MATERIALIZED VIEW LIVE.latest_articles_by_ticker
COMMENT "Latest 20 articles per ticker for dashboard"
AS
SELECT
  article_id,
  ticker,
  title,
  cleaned_text,
  source,
  url,
  published_at_ts,
  word_count,
  row_number() OVER (PARTITION BY ticker ORDER BY published_at_ts DESC) AS rn
FROM LIVE.articles_silver
WHERE rn <= 20;
