import json
import time
from google.cloud import pubsub_v1
from newsapi import NewsApiClient
import feedparser


class FinancialNewsPublisher:
    def __init__(self, project_id, topic_id):
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(project_id, topic_id)
        self.newsapi = NewsApiClient(api_key='YOUR_KEY')

    def fetch_newsapi(self):
        articles = self.newsapi.get_everything(
            q='stocks OR earnings OR fed OR inflation OR revenue',
            language='en',
            sort_by='publishedAt',
            page_size=100
        )
        return articles['articles']

    def fetch_rss(self, feeds):
        articles = []
        rss_feeds = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.investing.com/rss/news.rss',
            'https://finance.yahoo.com/news/rssindex'
        ]
        for url in rss_feeds:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                articles.append({
                    'title': entry.title,
                    'summary': entry.get('summary', ''),
                    'published': entry.get('published', ''),
                    'source': url
                })
        return articles

    def publish(self, article):
        data = json.dumps(article).encode('utf-8')
        future = self.publisher.publish(
            self.topic_path,
            data,
            source=article.get('source', 'unknown')
        )
        return future.result()

    def run(self, interval_seconds=300):
        while True:
            articles = self.fetch_newsapi()
            articles += self.fetch_rss([])
            for article in articles:
                self.publish(article)
            print(f"Published {len(articles)} articles")
            time.sleep(interval_seconds)


if __name__ == '__main__':
    publisher = FinancialNewsPublisher(
        project_id='finsentinel-nlp',
        topic_id='financial-news-stream'
    )
    publisher.run()
