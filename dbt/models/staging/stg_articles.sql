SELECT
    GENERATE_UUID() AS article_id,
    source,
    headline,
    cleaned_text,
    tickers,
    word_count,
    TIMESTAMP_TRUNC(published_at, HOUR) AS published_hour,
    DATE(published_at)                  AS published_date,
    ingested_at
FROM {{ source('finsentinel_raw', 'articles') }}
WHERE cleaned_text IS NOT NULL
  AND word_count >= 5
  AND word_count <= 512
