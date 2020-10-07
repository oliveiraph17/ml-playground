import pandas as pd
import spacy
import string

from datetime import datetime as dt
from joblib import dump, load
from spacy.lang.pt.stop_words import STOP_WORDS

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


stop_words = list(STOP_WORDS)
punctuation_characters = string.punctuation
persisted_model = True


# Custom transformer to be used in the first step of our Pipeline.
# It returns a stripped, lowercase version of each raw tweet passed in.
class data_cleaner(TransformerMixin):
    def transform(self, X):
        return [self.clean(text) for text in X]
    def fit(self, X, y, **fit_params):
        return self

    def clean(self, text):
        return text.strip().lower()


# Function used by the vectorizer.
def spacy_tokenizer(tweet):
    # Tokenizes tweet and lemmatizes the tokens.
    # It also gets rid of stop words and punctuation characters.
    doc = nlp(tweet)
    tokens = [token.lemma_ for token in doc]
    tokens = [token for token in tokens
              if token not in stop_words
              and token not in punctuation_characters]
    return tokens


if __name__ == '__main__':
    # spaCy is able to load this model because it was previously downloaded.
    nlp = spacy.load('pt_core_news_lg')

    df = pd.read_csv('../datasets/kaggle/training_data/50k.csv', sep=';')
    X = df['tweet_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.01,
                                                        random_state=17)

    data_cleaner = data_cleaner()
    try:
        vectorizer = load('vocabulary.joblib')
        classifier = load('fitted_multinomial_nb.joblib')
        pipeline = Pipeline([('data_cleaner', data_cleaner),
                             ('vectorizer', vectorizer),
                             ('classifier', classifier)])
    except OSError:
        print(f'{dt.now()} Could not load model. Creating a new one.')
        # The parameter to the tokenizer is a callable.
        vectorizer = CountVectorizer(tokenizer=spacy_tokenizer)
        classifier = MultinomialNB()
        # Pipeline for cleaning, vectorizing and classifying tweets.
        pipeline = Pipeline([('data_cleaner', data_cleaner),
                             ('vectorizer', vectorizer),
                             ('classifier', classifier)])
        print(f'{dt.now()} Fitting model...')
        pipeline.fit(X_train, y_train)
        if persisted_model:
            dump(vectorizer, 'vocabulary.joblib')
            dump(classifier, 'fitted_multinomial_nb.joblib')
    print(f'{dt.now()} Predicting...')
    preds = pipeline.predict(X_test)
    print(f'{dt.now()} Model accuracy: {pipeline.score(X_test, y_test)}')
