import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.io.gcp.bigquery import WriteToBigQuery
import json
import re
import langdetect


class CleanText(beam.DoFn):
    def process(self, element):
        article = json.loads(element.decode('utf-8'))
        text = article.get('title', '') + ' ' + article.get('summary', '')
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
        text = ' '.join(text.split())
        try:
            if langdetect.detect(text) != 'en':
                return
        except Exception:
            return
        article['cleaned_text'] = text
        article['word_count'] = len(text.split())
        yield article


class ExtractTickers(beam.DoFn):
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
               'META', 'NVDA', 'JPM', 'BAC', 'GS']

    def process(self, element):
        text = element.get('cleaned_text', '').upper()
        found = [t for t in self.TICKERS if t in text]
        element['tickers'] = found if found else ['MARKET']
        yield element


class DeduplicateDoFn(beam.DoFn):
    def __init__(self):
        self.seen = set()

    def process(self, element):
        key = hash(element.get('cleaned_text', '')[:100])
        if key not in self.seen:
            self.seen.add(key)
            yield element


BQ_SCHEMA = {
    'fields': [
        {'name': 'id',           'type': 'STRING'},
        {'name': 'source',       'type': 'STRING'},
        {'name': 'headline',     'type': 'STRING'},
        {'name': 'cleaned_text', 'type': 'STRING'},
        {'name': 'tickers',      'type': 'STRING', 'mode': 'REPEATED'},
        {'name': 'word_count',   'type': 'INTEGER'},
        {'name': 'published_at', 'type': 'TIMESTAMP'},
        {'name': 'ingested_at',  'type': 'TIMESTAMP'},
    ]
}


def run_pipeline():
    options = PipelineOptions(
        runner='DataflowRunner',
        project='finsentinel-nlp',
        region='us-central1',
        temp_location='gs://finsentinel-artifacts/temp',
        streaming=True
    )

    with beam.Pipeline(options=options) as p:
        (
            p
            | 'ReadFromPubSub' >> ReadFromPubSub(
                subscription='projects/finsentinel-nlp/subscriptions/news-processor'
              )
            | 'CleanText'      >> beam.ParDo(CleanText())
            | 'ExtractTickers' >> beam.ParDo(ExtractTickers())
            | 'Deduplicate'    >> beam.ParDo(DeduplicateDoFn())
            | 'WriteToBigQuery' >> WriteToBigQuery(
                'finsentinel-nlp:finsentinel_raw.articles',
                schema=BQ_SCHEMA,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
              )
        )


if __name__ == '__main__':
    run_pipeline()
