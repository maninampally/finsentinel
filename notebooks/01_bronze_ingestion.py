# Databricks notebook — run on Runtime 14.3 LTS ML cluster
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, current_timestamp, from_json, size, split
)
from pyspark.sql.types import StringType, StructField, StructType

spark = SparkSession.builder.appName('FinSentinel-Bronze').getOrCreate()

RAW_SCHEMA = StructType([
    StructField("title",     StringType()),
    StructField("summary",   StringType()),
    StructField("published", StringType()),
    StructField("source",    StringType()),
])

raw_stream = (
    spark.readStream
    .format('pubsub')
    .option('subscriptionId',
            'projects/finsentinel-nlp/subscriptions/news-processor')
    .option('credentialsFile', '/dbfs/finsentinel/creds.json')
    .load()
)

bronze_df = (
    raw_stream
    .select(from_json(col('data').cast('string'), RAW_SCHEMA).alias('payload'))
    .select('payload.*')
    .withColumn('ingested_at', current_timestamp())
    .withColumn('word_count',  size(split(col('title'), ' ')))
)

(
    bronze_df.writeStream
    .format('delta')
    .outputMode('append')
    .option('checkpointLocation', '/mnt/finsentinel/checkpoints/bronze')
    .toTable('finsentinel.bronze.raw_articles')
)
