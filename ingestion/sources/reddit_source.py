import praw


class RedditSource:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def fetch(self, subreddits: list = None, limit: int = 100) -> list:
        subreddits = subreddits or ['wallstreetbets', 'stocks', 'investing']
        articles = []
        for sub in subreddits:
            for post in self.reddit.subreddit(sub).hot(limit=limit):
                articles.append({
                    'title':     post.title,
                    'summary':   post.selftext[:500],
                    'published': str(post.created_utc),
                    'source':    f'reddit/r/{sub}'
                })
        return articles
