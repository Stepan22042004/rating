from joblib import load
import numpy as np
import pandas as pd
import gensim.downloader as api
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import os
import nltk

#nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for
             word in words if word not in stop_words]
    return ' '.join(words)


def get_vector(text, model):
    words = text.split()
    word_vecs = [model[word] for word in words if word in model]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(model.vector_size)


def predict(review):
    review = preprocess(review)
    model_path = os.path.join(os.path.dirname(__file__), 'my_word2vec.model')
    model = KeyedVectors.load(model_path)

    prediction = pd.DataFrame()
    prediction['vector'] = get_vector(review, model.wv)

    classifier_path = os.path.join(os.path.dirname(__file__), 'classifier_label.joblib')
    classifier = load(classifier_path)

    positive_path = os.path.join(os.path.dirname(__file__), 'regressor_positive.joblib')
    negative_path = os.path.join(os.path.dirname(__file__), 'regressor_negative.joblib')
    regressor_positive = load(positive_path)
    regressor_negative = load(negative_path)
    X = prediction['vector'].values

    prediction_label = classifier.predict(X.reshape(1, -1))

    if prediction_label == 1:
        prediction_rating = np.round(regressor_positive.predict(X.reshape(1, -1)))
    else:
        prediction_rating = np.round(regressor_negative.predict(X.reshape(1, -1)))

    prediction_rating = np.clip(prediction_rating, 1, 10)

    return prediction_label[0], prediction_rating[0]

