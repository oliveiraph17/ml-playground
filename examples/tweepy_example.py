from textblob import TextBlob
from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler
from tweepy.streaming import Stream
from tweepy.streaming import StreamListener

import numpy as np
import pandas as pd
import re

# File storing the user credentials to access the Twitter API.
import twitter_credentials


class TwitterAuthenticator():
    """
    Class for authenticating our requests to Twitter.
    """
    def authenticate_app(self):
        # The keys are unique identifiers that authenticate the app.
        auth = OAuthHandler(twitter_credentials.CONSUMER_KEY,
                            twitter_credentials.CONSUMER_KEY_SECRET)
        # The tokens allow the app to gain specific access to Twitter data.
        auth.set_access_token(twitter_credentials.ACCESS_TOKEN,
                              twitter_credentials.ACCESS_TOKEN_SECRET)
        return auth


class ScreenListener(StreamListener):
    """
    Basic listener class that prints out received tweets to the screen.
    """
    # Overriden method from StreamListener.
    # It can do whatever we want upon getting a tweet from StreamListener.
    # In this class, we will print out the tweets to the screen.
    def on_data(self, raw_data):
        print(raw_data)
        return True

    # Overriden method from StreamListener.
    # It can do whatever we want upon getting an error from StreamListener.
    # In this class, we will print out the error status code to the screen.
    def on_error(self, status_code):
        if status_code in [420, 429]:
            print('Reached rate limit. Disconnecting the stream.')
            return False
        else:
            print(f'Error: {status_code}.')
            return True


class TwitterStreamer():
    """
    Class for streaming live tweets.
    """
    def __init__(self):
        self.authenticator = TwitterAuthenticator()

    # Handles Twitter authentication and fetches tweets.
    def stream_tweets(self, keywords, languages=None):
        listener = ScreenListener()
        auth = self.authenticator.authenticate_app()
        stream = Stream(auth, listener)

        # Tweets are filtered by keyword and language.
        stream.filter(track=keywords, languages=languages)


class TwitterClient():
    """
    Class for browsing through tweets using pagination via the Twitter API.
    """
    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_app()
        self.api = API(self.auth)
        self.user = twitter_user

    def get_user_timeline_tweets(self, num_tweets):
        tweets_list = []
        cursor = Cursor(self.api.user_timeline, id=self.user)
        for tweet in cursor.items(num_tweets):
            tweets_list.append(tweet)
        return tweets_list

    def get_api(self):
        return self.api


class TweetAnalyzer():
    """
    Class for analyzing and categorizing content from tweets.
    """
    def clean_tweet(self, raw_tweet):
        # Patterns consisting of 3 capturing groups:
        # 1) user mentions;
        # 2) characters that are not alphanumeric nor spaces/tabs; and
        # 3) URLs.
        patterns = '(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)'
        # Anything captured by these patterns is removed from the tweet.
        return re.sub(patterns, '', raw_tweet)

    def analyze_sentiment(self, raw_tweet):
        analysis = TextBlob(self.clean_tweet(raw_tweet))
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1

    def tweets_to_dataframe(self, raw_tweets):
        # The result set of a Tweepy response is based on the Python list.
        # Each element in the result set is of Status type.
        # We use dot notation to access each property in a Status element.
        text_list = [t.text for t in raw_tweets]
        df = pd.DataFrame(data=text_list, columns=['_tweet_'])
        df['_lang_'] = np.array([t.lang for t in raw_tweets])
        df['_date_'] = np.array([t.created_at for t in raw_tweets])
        df['_likes_'] = np.array([t.favorite_count for t in raw_tweets])
        return df


if __name__ == '__main__':
    twitter_client = TwitterClient()
    tweet_analyzer = TweetAnalyzer()
    api = twitter_client.get_api()
    tweets = api.user_timeline(screen_name='realDonaldTrump', count=3)
    df = tweet_analyzer.tweets_to_dataframe(tweets)
    sentiment_list = [tweet_analyzer.analyze_sentiment(t)
                      for t in df['_tweet_']]
    df['sentiment'] = np.array(sentiment_list)
    print(df)

    streamer = TwitterStreamer()
    streamer.stream_tweets(['switch', 'ps5', 'xbox'], ['pt'])
