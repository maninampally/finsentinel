SELECT
    id AS article_id,
    title,
    summary AS cleaned_text,
    content,
    author,
    source,
    url,
    CAST(published AS TIMESTAMP) AS published_at,
    CAST(retrieved_at AS TIMESTAMP) AS ingested_at
FROM {{ source('finsentinel_silver', 'articles') }}
WHERE summary IS NOT NULL
  AND LENGTH(summary) >= 20
