import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import gensim.downloader as api
from joblib import dump
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from gensim.models import KeyedVectors

#nltk.download('punkt_tab')

def get_vector(text, model):
    words = text.split()
    word_vecs = [model[word] for word in words if word in model]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(model.vector_size)


train_data = pd.read_csv('data.csv')

"""
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in train_data['review']]

# Обучение модели Word2Vec
model = Word2Vec(sentences=tokenized_sentences, vector_size=300, window=10, min_count=3, workers=4, sg=1, epochs=5)

# Сохранение модели
model.save("my_word2vec.model")

"""
model_path = os.path.join(os.path.dirname(__file__), 'my_word2vec.model')
model = KeyedVectors.load(model_path)

test_data = pd.read_csv('test.csv')

label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
test_data['label'] = label_encoder.fit_transform(test_data['label'])


train_data['vectors'] = train_data['cleaned_review'].apply(lambda text: get_vector(text, model.wv))
test_data['vectors'] = test_data['cleaned_review'].apply(lambda text: get_vector(text, model.wv))


positive_data = train_data[train_data['label'] == 1]
negative_data = train_data[train_data['label'] == 0]

positive_data_test = test_data[test_data['label'] == 1]
negative_data_test = test_data[test_data['label'] == 0]

X_train_all = train_data['vectors'].to_list()
X_train_pos = positive_data['vectors'].to_list()
X_train_neg = negative_data['vectors'].to_list()
y_train_ratings_pos = positive_data['rating'].to_numpy()
y_train_ratings_neg = negative_data['rating'].to_numpy()
y_train_labels = train_data['label'].to_numpy()
X_test_all = test_data['vectors'].to_list()
y_test_labels = test_data['label'].to_numpy()
y_test_ratings = test_data['rating'].to_numpy()


classifier = LogisticRegression(solver='liblinear', C=1.7)
classifier.fit(X_train_all, y_train_labels)
dump(classifier, 'classifier_label.joblib')

predictions_labels = classifier.predict(X_test_all)

predicted = test_data
predicted['predictions_labels'] = predictions_labels
X_test = predicted['vectors'].to_numpy()

accuracy = accuracy_score(y_test_labels, predictions_labels)

regressor_positive = LinearRegression()
regressor_positive.fit(X_train_pos, y_train_ratings_pos)
dump(regressor_positive, 'regressor_positive.joblib')

regressor_negative = LinearRegression()
regressor_negative.fit(X_train_neg, y_train_ratings_neg)
dump(regressor_negative, 'regressor_negative.joblib')

predictions_ratings = []

for i in range(len(test_data)):
    if predicted.loc[i, 'predictions_labels'] == 1:
        # Предсказание рейтинга для позитивных
        predictions_ratings.append(np.round(regressor_positive.predict(X_test[i].reshape(1, -1))))
    else:
        # Предсказание рейтинга для негативных
        predictions_ratings.append(np.round(regressor_negative.predict(X_test[i].reshape(1, -1))))


mae = mean_absolute_error(y_test_ratings, predictions_ratings)

print(f'Mean_absolute_error: {mae}')
print(f'Accuracy: {accuracy}')


