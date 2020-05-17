import os
import textstat
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
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# pwd

# cd'/Users/ericross/School/Queens_MMAI/MMAI/MMAI_891/Project'

# Directories
DATASET_DIR = './asap-aes/'
WORKBOOK_DIR = './courses/'

# Import Data
df = pd.read_csv(os.path.join(DATASET_DIR, "training_set_rel3.tsv"), sep='\t', encoding='ISO-8859-1')
df.info()
df.head()

df1 = pd.DataFrame(df['essay'])
df1 = df1.iloc[12253:]
print(df1.iloc[0])
df1.head()

print(df1.shape)

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


df1['essay_pre'] = df1['essay'].apply(preprocess)

print(df1.iloc[0, :].essay)
print(df1.iloc[0, :].essay_pre)

df1 = pd.DataFrame(df1['essay_pre'])

vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.05, max_features=500, ngram_range=[1,3])
dtm = vectorizer.fit_transform(df1['essay_pre'])

vectorizer.get_feature_names()

bow_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names(),index=df1.index)

df1_bow = pd.concat([df1, bow_df], axis=1)
print(df1_bow.shape)

# Feature Engineering
df1_bow['length'] = df1_bow['essay_pre'].apply(lambda x: len(x))
df1_bow['syllable count'] = df1_bow['essay_pre'].apply(lambda x: textstat.syllable_count(x))
df1_bow['flesch_reading_ease'] = df1_bow['essay_pre'].apply(lambda x: textstat.flesch_reading_ease(x))
df1_bow = df1_bow.drop(['essay_pre'], axis=1)

# Split X & Y From Dataframe
y = df['domain1_score']
y = y.iloc[12253:]
X = df1_bow

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Creation
regr = LinearRegression()
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

