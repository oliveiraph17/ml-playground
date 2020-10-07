import matplotlib.pyplot as plt
import pandas as pd
import spacy
import streamlit as st
import string

from joblib import dump, load
from spacy.lang.pt.stop_words import STOP_WORDS

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler

# File storing the user credentials to access the Twitter API.
import my_twitter_credentials


VOCABULARY_FILENAME = 'better_learned_vocabulary.joblib'
MODEL_FILENAME = 'better_fitted_multinomial_nb.joblib'


###############################################################################
# Code related to Tweepy.
###############################################################################
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


class TwitterClient():
    """
    Class for browsing through tweets via the Twitter API.
    """
    def __init__(self):
        self.auth = TwitterAuthenticator().authenticate_app()
        self.api = API(self.auth)

    def search_tweet(self, query_string):
        cursor = Cursor(self.api.search, q=query_string, lang='pt', count=5)
        return cursor.items(5)
###############################################################################


###############################################################################
# Code related to spaCy.
###############################################################################
nlp = spacy.load('pt_core_news_sm')
STOP_WORDS_LIST = list(STOP_WORDS)
PUNCTUATION_CHARACTERS_LIST = list(string.punctuation)


# Tokenizer used by the CountVectorizer.
def spacy_tokenizer(tweet):
    # Tokenizes tweet and lemmatizes the tokens.
    # It also gets rid of stop words and punctuation characters.
    doc = nlp(tweet)
    tokens = [token.lemma_ for token in doc]
    tokens = [token for token in tokens
              if token not in STOP_WORDS_LIST
              and token not in PUNCTUATION_CHARACTERS_LIST]
    return tokens
###############################################################################


###############################################################################
# Execution code.
###############################################################################
def get_training_data():
    df = pd.read_csv('../datasets/kaggle/training_data/50k.csv', sep=';')
    X = df['tweet_text']
    y = df['sentiment']
    return df, train_test_split(X, y, test_size=0.01, random_state=17)

def execute_pipeline(X_train, X_test, y_train, y_test):
    loaded = False
    try:
        vectorizer = load(VOCABULARY_FILENAME)
        classifier = load(MODEL_FILENAME)
        pipeline = Pipeline([('vectorizer', vectorizer),
                             ('classifier', classifier)], verbose=True)
        loaded = True
    except OSError:
        # The parameter to the tokenizer is a callable.
        vectorizer = CountVectorizer(tokenizer=spacy_tokenizer)
        # vectorizer = CountVectorizer(tokenizer=spacy_tokenizer)
        classifier = MultinomialNB()
        # Pipeline for cleaning, vectorizing and classifying tweets.
        pipeline = Pipeline([('vectorizer', vectorizer),
                             ('classifier', classifier)], verbose=True)
        status = st.text('[STATUS] Fitting model...')
        pipeline.fit(X_train, y_train)
        status.text('[STATUS] Model was fit.')
        dump(vectorizer, VOCABULARY_FILENAME)
        dump(classifier, MODEL_FILENAME)
    return loaded, pipeline

def plot_chart(input_df):
    input_df['tweet_date'] = pd.to_datetime(input_df['tweet_date'])
    input_df.sort_values(by=['tweet_date'], inplace=True)

    positive_df = input_df[input_df['sentiment'] == 1]
    positive_df['month'] = positive_df['tweet_date'].map(lambda d: d.month)
    positive_df['day'] = positive_df['tweet_date'].map(lambda d: d.day)
    agg_positive_df = positive_df.groupby(['month', 'day']).count()
    positive_counts = agg_positive_df['id'].tolist()

    negative_df = input_df[input_df['sentiment'] == 0]
    negative_df['month'] = negative_df['tweet_date'].map(lambda d: d.month)
    negative_df['day'] = negative_df['tweet_date'].map(lambda d: d.day)
    agg_negative_df = negative_df.groupby(['month', 'day']).count()
    negative_counts = agg_negative_df['id'].tolist()

    x_labels = pd.date_range(start='09/28/2018', end='10/12/2018')
    x_labels = [str(d).split()[0] for d in x_labels.to_series().tolist()]

    positive_dates = [
        '2018-09-28', '2018-09-30', '2018-10-01',
        '2018-10-02', '2018-10-03', '2018-10-05',
        '2018-10-06', '2018-10-08', '2018-10-12'
    ]
    negative_dates = [
        '2018-10-01', '2018-10-02', '2018-10-08'
    ]

    positive_y_values = [0] * len(x_labels)
    negative_y_values = [0] * len(x_labels)
    for i in range(len(x_labels)):
        if x_labels[i] in positive_dates:
            positive_y_values[i] = positive_counts.pop(0)
        if x_labels[i] in negative_dates:
            negative_y_values[i] = negative_counts.pop(0)

    plt_fig, ax = plt.subplots()
    ax.plot(range(len(x_labels)), positive_y_values,
            label='Positive', color='#4287f5')
    ax.plot(range(len(x_labels)), negative_y_values,
            label='Negative', color='#f54242')
    ax.set_title('Tweet sentiment evolution of training set')
    ax.set_xlabel('Day')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation='vertical')
    ax.set_ylabel('Number of tweets')
    ax.legend()
    st.pyplot(fig=plt_fig)


if __name__ == '__main__':
    st.title('Twitter App')
    client = TwitterClient()

    df, (X_train, X_test, Y_train, Y_test) = get_training_data()
    loaded, pipeline = execute_pipeline(X_train, X_test, Y_train, Y_test)
    if loaded:
        st.write('Vocabulary and model successfully loaded.')
    st.text(f'Model accuracy on test data: {pipeline.score(X_test, Y_test)}')

    theme_options = ('Weather', 'Politics')
    theme = st.radio("Pick your theme to search for a tweet.", theme_options)
    if theme == 'Weather':
        keywords = 'calor OR quente OR aquecimento OR queimadas'
    else:
        keywords = 'bolsonaro OR mito OR lula OR pt OR trump'

    # Shows tweets with predictions.
    count = 0
    tweets = client.search_tweet(keywords)
    for tweet in tweets:
        area_key = 'a' + str(count)
        st.text_area('Fetched tweet', value=tweet.text, key=area_key)
        count += 1

        prediction = pipeline.predict([tweet.text])
        if prediction[0] == 0:
            st.info('Prediction: negative')
        else:
            st.info('Prediction: positive')

    # Shows plot.
    plot_chart(df)
###############################################################################
