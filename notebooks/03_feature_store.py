# Databricks notebook — Feature Store registration
from databricks.feature_store import FeatureStoreClient, FeatureLookup

fs = FeatureStoreClient()

gold_df = spark.table('finsentinel.gold.sentiment_features')

# First run only — create the Feature Store table
fs.create_table(
    name='finsentinel.features.ticker_sentiment',
    primary_keys=['ticker', 'window_start'],
    timestamp_keys=['window_start'],
    df=gold_df,
    schema=gold_df.schema,
    description='Per-ticker hourly sentiment aggregations with point-in-time support'
)

# Subsequent runs — write/merge features
fs.write_table(
    name='finsentinel.features.ticker_sentiment',
    df=gold_df,
    mode='merge'
)

# Point-in-time correct feature lookup for training
# Eliminates lookahead bias — critical for financial ML
training_set = fs.create_training_set(
    df=labels_df,
    feature_lookups=[
        FeatureLookup(
            table_name='finsentinel.features.ticker_sentiment',
            feature_names=['article_count', 'texts'],
            lookup_key=['ticker'],
            timestamp_lookup_key='event_timestamp'
        )
    ],
    label='sentiment_label'
)

train_df = training_set.load_dataframe().toPandas()
