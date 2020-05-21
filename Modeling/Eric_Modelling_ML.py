import os
import textstat
import nltk
import re
import collections
from collections import defaultdict
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
from Modeling.final_preprocessing import lem, pre_process, count_spell_error

os.getcwd()

os.chdir('/Users/ericross/School/Queens_MMAI/MMAI/MMAI_891/Project')

# Directories
DATASET_DIR = './asap-aes/'
WORKBOOK_DIR = './courses/'

# Import Data For Mispelled Word Counter
df1 = pd.read_csv(os.path.join(DATASET_DIR, "training_set_rel3.tsv"), sep='\t', encoding='ISO-8859-1')
df1 = df1[12253:].copy()
df1 = pd.DataFrame(data=df1, columns=["essay","domain1_score"])
df1 = df1.reset_index(drop=True)

# Import Data from Spell Checked Document
df = pd.read_csv(os.path.join(DATASET_DIR, "Essays_SpellCheck_Set8.csv"))
df = df.drop(['Unnamed: 0'], axis=1)
df.info()
df.head()

# stop_words = set(stopwords.words('english') + stopwords.word('french'))   # haven't used this below


# Delete the essays that have less than 200 words ~ approx. 20 essays --> need to figure this out

# df['len_words'] = df['Essay_Prep'].apply(nltk.word_tokenize(['Essay_Prep']))
# less_than_200_index = df[ df['len_words']].index
# df.drop(less_than_200_index, inplace=True


# Apply pre_processing onto both dataframes
df['Essay_Prep'] = df['Essay'].apply(pre_process)
df1['Essay_Prep'] = df1['essay'].apply(pre_process)

print(df.shape, df1.shape)


# Apply the spell checker on df1
# df1['Spelling_Mistakes_Count'] = df1['Essay_Prep'].apply(count_spell_error)


# Apply the lemmatizer on df
df['Essay_Prep'] = df['Essay_Prep'].apply(lem)


# Vectorizing the text and converting to columns (taken from Steve's Session 5)

vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.05, max_features=500, ngram_range=[1, 3])
dtm = vectorizer.fit_transform(df['Essay_Prep'])

vectorizer.get_feature_names()

bag_of_word_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names(), index=df1.index)

df_bow = pd.concat([df, bag_of_word_df], axis=1)
print(df_bow.shape)

# Feature Engineering (#Taken from Steve's Lecture 5)
df_bow['Length'] = df_bow['Essay_Prep'].apply(lambda x: len(x))
df_bow['Syllable_Count'] = df_bow['Essay_Prep'].apply(lambda x: textstat.syllable_count(x))
df_bow['Flesch_Reading_Ease'] = df_bow['Essay_Prep'].apply(lambda x: textstat.flesch_reading_ease(x))
df_bow = df_bow.drop(['Essay_Prep', 'Essay'], axis=1)

# Check for Mispelled Words in Original DataFrame (df1) --> can't seem to get this to work yet
# could be because I'm trying to append a function a new dataframe
# Do we need to run the spellchecker after some basic cleaning

#df1['Spelling_Mistakes_Count'] = df1['Essay_Prep'].apply(count_spell_error)
#print(df1.shape)


# Split X & Y From Dataframe
X = df_bow.drop(['Essay Score'], axis=1)
X = X.values
y = df_bow['Essay Score'].values

print(y.shape, type(y), X.shape, type(X))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


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
