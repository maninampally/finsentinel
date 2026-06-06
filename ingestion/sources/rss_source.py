import feedparser

RSS_FEEDS = [
    'https://feeds.bloomberg.com/markets/news.rss',
    'https://www.investing.com/rss/news.rss',
    'https://finance.yahoo.com/news/rssindex',
]


class RSSSource:
    def __init__(self, feeds: list = None):
        self.feeds = feeds or RSS_FEEDS

    def fetch(self) -> list:
        articles = []
        for url in self.feeds:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                articles.append({
                    'title':     entry.get('title', ''),
                    'summary':   entry.get('summary', ''),
                    'published': entry.get('published', ''),
                    'source':    url
                })
        return articles
