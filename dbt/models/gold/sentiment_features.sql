WITH base AS (
    SELECT * FROM {{ ref('stg_articles') }}
),
ticker_exploded AS (
    SELECT
        article_id,
        cleaned_text,
        ticker,
        published_hour,
        published_date
    FROM base,
    UNNEST(tickers) AS ticker
)
SELECT
    ticker,
    published_date,
    published_hour,
    COUNT(*)                                              AS article_count,
    ARRAY_AGG(cleaned_text ORDER BY published_hour)      AS texts,
    CURRENT_TIMESTAMP()                                   AS updated_at
FROM ticker_exploded
GROUP BY ticker, published_date, published_hour
