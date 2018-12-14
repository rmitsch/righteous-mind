# See https://towardsdatascience.com/how-to-use-twitters-api-c3e25c2ad8fe.
import twitter
import re
import datetime
import pandas as pd


class TwitterMiner:
    """
    Wrapper for mining user timelines.
    """

    def __init__(self, request_limit=20):
        self.request_limit = request_limit
        self._twitter_keys = {
            'consumer_key': "",  # add your consumer key
            'consumer_secret': "",  # add your consumer secret key
            'access_token_key': "",  # add your access token key
            'access_token_secret': "" # add your access token secret key
        }
        self.api = self.init_api()

    def init_api(self) -> twitter.Api:
        """
        Initializes twitter API client.
        :return:
        """
        return twitter.Api(
            consumer_key=self._twitter_keys['consumer_key'],
            consumer_secret=self._twitter_keys['consumer_secret'],
            access_token_key=self._twitter_keys['access_token_key'],
            access_token_secret=self._twitter_keys['access_token_secret']
        )

    def mine_user_tweets(self, user=" set default user to get data from") -> pd.DataFrame:
        """
        Fetches as many user tweets from timeline as possible.
        :param user:
        :return:
        """
        statuses = self.api.GetUserTimeline(screen_name=user, count=self.request_limit)
        data = []

        for item in statuses:
            data.append({
                'tweet_id': item.id,
                'handle': item.user.name,
                'retweet_count': item.retweet_count,
                'text': item.text,
                'mined_at': datetime.datetime.now(),
                'created_at': item.created_at,
            })

        return pd.DataFrame(data)
