import json
import os
import time

from google.cloud import pubsub_v1

from ingestion.sources.newsapi_source import NewsAPISource
from ingestion.sources.reddit_source import RedditSource
from ingestion.sources.rss_source import RSSSource


class FinancialNewsPublisher:
    def __init__(self, project_id: str, topic_id: str):
        self.publisher  = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_id)

        self._newsapi = NewsAPISource(api_key=os.environ['NEWSAPI_KEY'])
        self._rss     = RSSSource()

        reddit_id     = os.environ.get('REDDIT_CLIENT_ID')
        reddit_secret = os.environ.get('REDDIT_CLIENT_SECRET')
        self._reddit  = (
            RedditSource(
                client_id=reddit_id,
                client_secret=reddit_secret,
                user_agent='finsentinel/1.0'
            )
            if reddit_id and reddit_secret else None
        )

    def fetch_all(self) -> list:
        articles = self._newsapi.fetch() + self._rss.fetch()
        if self._reddit:
            articles += self._reddit.fetch()
        return articles

    def publish(self, article: dict):
        data = json.dumps(article).encode('utf-8')
        future = self.publisher.publish(
            self.topic_path,
            data,
            source=article.get('source', 'unknown')
        )
        future.result()

    def run(self, interval_seconds: int = 300):
        while True:
            articles = self.fetch_all()
            for article in articles:
                self.publish(article)
            print(f"Published {len(articles)} articles")
            time.sleep(interval_seconds)


if __name__ == '__main__':
    publisher = FinancialNewsPublisher(
        project_id=os.environ['GCP_PROJECT_ID'],
        topic_id=os.environ.get('PUBSUB_TOPIC', 'financial-news-stream')
    )
    publisher.run()
