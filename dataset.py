import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd


# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for
             word in words if word not in stop_words]
    return ' '.join(words)


def get_data(path, label=None):
    files = os.listdir(path)
    X = []
    y_rating = []
    for file in files:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            rev = f.read()
            X.append(rev)
            rating = int(file.split('_')[1].split('.')[0])
            y_rating.append(rating)
    y_label = [label] * len(X)
    return X, y_label, y_rating

# если изменить путь, то можно создать csv с тестовым датасетом


positive = './aclimdb/train/pos'
negative = './aclimdb/train/neg'
X, y_label, y_rating = get_data(path=positive, label='pos')
X_neg, y_label_neg, y_rating_neg = get_data(path=negative, label='neg')
X += X_neg
y_label += y_label_neg
y_rating += y_rating_neg

data = pd.DataFrame()
data['label'] = y_label
data['rating'] = y_rating
data['review'] = X
data['cleaned_review'] = data['review'].apply(preprocess)

data.to_csv('data.csv')
