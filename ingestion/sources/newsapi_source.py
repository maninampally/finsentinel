from newsapi import NewsApiClient


class NewsAPISource:
    def __init__(self, api_key: str):
        self.client = NewsApiClient(api_key=api_key)

    def fetch(self, query: str = 'stocks OR earnings OR fed OR inflation OR revenue',
              page_size: int = 100) -> list:
        response = self.client.get_everything(
            q=query,
            language='en',
            sort_by='publishedAt',
            page_size=page_size
        )
        return response.get('articles', [])
