Data Science Twitter App
========================

This app is available as a [Docker image][1].  
If you download and run the Docker image, you should be able to use the app at **http://localhost:8501**.  
Alternatively, you can clone this repository, enter the twitter_app folder, and do the following:
* run the command ``pip install -r requirements.txt`` to install the app dependencies; and
* run the command ``streamlit run twitter_app.py`` to execute the app and use it.

Among the libraries used, these are worth mentioning:
* [Tweepy][3], used for accessing the Twitter API;
* [spaCy][4], which is an NLP library that I used for tokenizing and lemmatizing tweets; and
* [Streamlit][5], which is a framework for creating ML apps with a GUI.

I built my own tokenizer based on a spaCy model called [pt_core_news_sm][6].  
It is a CNN trained on Portuguese corpora.  
The tokenizer was used by the [CountVectorizer][7] module from the **sklearn** library.  
Using the module's default tokenizer had enabled around 74% of accuracy when predicting the test data.  
On the other hand, using my own tokenizer enabled around 98% of accuracy when predicting the same test data.  
Hence, building my own tokenizer paid off.  
Speaking of which, accuracy was the chosen metric for reporting model performance.  
This is because I was interested in assessing how well the model was doing at predicting unseen test data.

The classifier employed was [Multinomial Naive Bayes][8], which works nicely with features based on word counts.  
I decided to use the model with its default parameters, without performing a random or grid search.  
Moreover, the better performance enabled by my own tokenizer was in detriment of execution time.  
Specifically, it took around 5 minutes to tokenize the training set, as opposed to the default tokenizer, which executes instantly.  
This is why I decided to speed up the training process by not running any cross-validation procedure.

The training data was [this file][2], which has 50k labelled tweets in Portuguese.  
The data are balanced and have 2 classes: *positive* and *negative*.  
Specifically, the app uses 99% of the content for training and 1% for testing the model.

The source code makes use of a [Pipeline][9], which improves code readability and maintainability.  
It is useful when doing repetitive tasks involving data transformation and prediction.

The app allows you to pick a topic for tweets.  
Then, it analyzes the sentiment of 5 retrieved tweets.  
It is possible to fetch more tweets either by choosing another topic or by refreshing the app at the top-right corner.  
Furthermore, the app plots a chart depicting the tweet sentiment evolution of the training data.

[1]: https://hub.docker.com/r/oliveiraph17/data-science-twitter-app
[2]: https://github.com/oliveiraph17/ml-playground/blob/main/twitter_app/50k.csv
[3]: https://www.tweepy.org/
[4]: https://spacy.io/
[5]: https://www.streamlit.io/
[6]: https://spacy.io/models/pt#pt_core_news_sm
[7]: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
[8]: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
[9]: http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
