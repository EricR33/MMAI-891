import os
# import nltk
import re
import unidecode
# import numpy as np
import pandas as pd
# import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# pwd

# cd'/Users/ericross/School/Queens_MMAI/MMAI/MMAI_891/Project'

# Directories
DATASET_DIR = './asap-aes/'
WORKBOOK_DIR = './courses/'

# Import Data
df = pd.read_csv(os.path.join(DATASET_DIR, "training_set_rel3.tsv"), sep='\t', encoding='ISO-8859-1')

df.info()
df.head()

# Split X & Y
X = pd.DataFrame(df['essay'])
X = X.iloc[12253:]
print(X.iloc[0])
X.head()

y = df['domain1_score']
y = y.iloc[12253:]
y.head()

print(y.shape, X.shape)

# stop_words = set(stopwords.words('english') + stopwords.word('french'))   # haven't used this below

lemmer = WordNetLemmatizer()

# Regular Expression Cleaning


def preprocess(x):
    # x = re.sub(r"[abc\\]$", "'", x)        # trying to get rid of \'s notation
    x = re.sub(r"@CAPS", " ", x)
    x = re.sub(r"@DATE", " ", x)
    x = re.sub(r"@LOCATION", " ", x)
    x = re.sub(r"@ORGANIZATION", " ", x)
    x = re.sub(r"@NUM", " ", x)
    x = re.sub(r"@PERCENT", " ", x)
    x = re.sub(r"@PERSON", " ", x)
    x = re.sub(r"@MONTH", " ", x)
    x = re.sub(r"@CITY", " ", x)
    x = re.sub(r"@YEAR", " ", x)
    x = re.sub(r'[^\w\s]', '', x)
    x = unidecode.unidecode(x)
    x = re.sub(r'\s+', ' ', x)
    x = re.sub(r'\d+', '', x)
    x = x.lower()
    x = re.sub(r"that's", "that is", x)
    x = re.sub(r"there's", "there is", x)
    x = re.sub(r"what's", "what is", x)
    x = re.sub(r"where's", "where is", x)
    x = re.sub(r"it's", "it is", x)
    x = re.sub(r"who's", "who is", x)
    x = re.sub(r"i'm", "i am", x)
    x = re.sub(r"she's", "she is", x)
    x = re.sub(r"he's", "he is", x)
    x = re.sub(r"they're", "they are", x)
    x = re.sub(r"who're", "who are", x)
    x = re.sub(r"you're", "you are", x)
    x = re.sub(r"ain't", "am not", x)
    x = re.sub(r"aren't", "are not", x)
    x = re.sub(r"wouldn't", "would not", x)
    x = re.sub(r"shouldn't", "should not", x)
    x = re.sub(r"couldn't", "could not", x)
    x = re.sub(r"doesn't", "does not", x)
    x = re.sub(r"isn't", "is not", x)
    x = re.sub(r"can't", "can not", x)
    x = re.sub(r"couldn't", "could not", x)
    x = re.sub(r"won't", "will not", x)
    x = re.sub(r"i've", "i have", x)
    x = re.sub(r"you've", "you have", x)
    x = re.sub(r"they've", "they have", x)
    x = re.sub(r"we've", "we have", x)
    x = re.sub(r"don't", "do not", x)
    x = re.sub(r"didn't", "did not", x)
    x = re.sub(r"i'll", "i will", x)
    x = re.sub(r"you'll", "you will", x)
    x = re.sub(r"he'll", "he will", x)
    x = re.sub(r"she'll", "she will", x)
    x = re.sub(r"they'll", "they will", x)
    x = re.sub(r"we'll", "we will", x)
    x = re.sub(r"i'd", "i would", x)
    x = re.sub(r"you'd", "you would", x)
    x = re.sub(r"he'd", "he would", x)
    x = re.sub(r"she'd", "she would", x)
    x = re.sub(r"they'd", "they would", x)
    x = re.sub(r"we'd", "we would", x)
    x = re.sub(r"she's", "she has", x)
    x = re.sub(r"he's", "he has", x)
    x = [lemmer.lemmatize(w) for w in x.split()]
    return ' '.join(x)


X['essay_pre'] = X['essay'].apply(preprocess)

print(X.iloc[0, :].essay)
print(X.iloc[0, :].essay_pre)

X = pd.DataFrame(X['essay_pre'])

vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.05, max_features=500, ngram_range=[1,3])
dtm = vectorizer.fit_transform(X['essay_pre'])

