import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from ingestion.sources.rss_source import RSSSource
import pandas as pd


BRONZE_DIR = Path(os.getenv('FINSENTINEL_BRONZE', 'data/bronze'))


def _fingerprint(record: dict) -> str:
    key = (record.get('title', '') or '') + '|' + (record.get('source', '') or '') + '|' + (record.get('published', '') or '')
    return hashlib.sha1(key.encode('utf-8')).hexdigest()


def _normalize_article(article: dict) -> dict:
    # canonical fields: id, title, summary, published, source, url, retrieved_at, language, raw
    published = article.get('published') or article.get('publishedAt') or ''
    try:
        published_ts = pd.to_datetime(published, utc=True)
        published_iso = published_ts.isoformat()
    except Exception:
        published_iso = ''

    normalized = {
        'title': article.get('title') or article.get('headline') or '',
        'summary': article.get('summary') or article.get('description') or article.get('content') or '',
        'published': published_iso,
        'source': article.get('source') if isinstance(article.get('source'), str) else (article.get('source', {}).get('name') if isinstance(article.get('source'), dict) else ''),
        'url': article.get('url') or article.get('link') or '',
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'language': article.get('language', 'en'),
        'raw': article,
    }
    normalized['id'] = _fingerprint(normalized)
    return normalized


def write_ndjson(records: List[dict]):
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    out_file = BRONZE_DIR / f'{date_str}.ndjson'
    # append
    with out_file.open('a', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def ingest_once() -> int:
    # load keys from env
    newsapi_key = os.getenv('NEWSAPI_KEY', '')
    reddit_client_id = os.getenv('REDDIT_CLIENT_ID', '')
    reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
    reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'finsentinel/1.0')

    sources = []
    if newsapi_key:
        try:
            from ingestion.sources.newsapi_source import NewsAPISource
            sources.append(NewsAPISource(api_key=newsapi_key))
        except ModuleNotFoundError as exc:
            print(f'Skipping NewsAPI source: {exc}')
    sources.append(RSSSource())
    if reddit_client_id and reddit_client_secret:
        try:
            from ingestion.sources.reddit_source import RedditSource
            sources.append(RedditSource(client_id=reddit_client_id, client_secret=reddit_client_secret, user_agent=reddit_user_agent))
        except ModuleNotFoundError as exc:
            print(f'Skipping Reddit source: {exc}')

    all_articles = []
    for src in sources:
        try:
            if hasattr(src, 'fetch'):
                all_articles.extend(src.fetch())
        except Exception as e:
            print(f'Error fetching from source {src}: {e}')

    normalized = []
    seen = set()
    for a in all_articles:
        n = _normalize_article(a)
        if n['id'] in seen:
            continue
        seen.add(n['id'])
        normalized.append(n)

    if normalized:
        write_ndjson(normalized)

    print(f'Ingested {len(normalized)} records')
    return len(normalized)


if __name__ == '__main__':
    ingest_once()
