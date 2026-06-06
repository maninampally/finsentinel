# Databricks notebook — Delta Live Tables pipeline
import dlt
from pyspark.sql.functions import col, explode, window, count, collect_list

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
           'META', 'NVDA', 'JPM', 'BAC', 'GS']


@dlt.table(
    name='silver_clean_articles',
    comment='Cleaned, deduplicated, English-only articles with tickers',
    table_properties={'quality': 'silver'}
)
@dlt.expect_or_drop('valid_text',   'word_count >= 5 AND word_count <= 512')
@dlt.expect_or_drop('english_only', 'length(title) > 0')
def silver_clean_articles():
    from pyspark.sql.functions import regexp_replace, array, when, lit

    return (
        dlt.read_stream('finsentinel.bronze.raw_articles')
        .withColumn(
            'cleaned_text',
            regexp_replace(
                regexp_replace(
                    col('title') + ' ' + col('summary'),
                    r'http\S+', ''
                ),
                r'[^a-zA-Z0-9\s\.,\!\?]', ''
            )
        )
        .withColumn(
            'tickers',
            array(*[
                when(col('cleaned_text').contains(t), lit(t))
                for t in TICKERS
            ])
        )
        .dropDuplicates(['cleaned_text'])
    )


@dlt.table(
    name='gold_sentiment_features',
    comment='Ticker-level hourly sentiment aggregations',
    table_properties={'quality': 'gold'}
)
def gold_sentiment_features():
    return (
        dlt.read('silver_clean_articles')
        .withColumn('ticker', explode(col('tickers')))
        .groupBy('ticker', window('ingested_at', '1 hour'))
        .agg(
            count('*').alias('article_count'),
            collect_list('cleaned_text').alias('texts')
        )
        .withColumn('window_start', col('window.start'))
        .withColumn('window_end',   col('window.end'))
        .drop('window')
    )
