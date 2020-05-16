import os
# import nltk
import re
import unidecode
# import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import SpatialDropout1D, LSTM, Dense
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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

# Finding the maximum words for an essay --> Can't get this work: It only works with series type pandas data
# maximum = 0
# for i in range(len(X)):
#    if len(nltk.word_tokenize(X.iloc[i])) > maximum:
#        maximum = len(nltk.word_tokenize(X.iloc[i]))
# print(maximum)

# Finding the minimum words for an essay
# minimum = 1051
# for row in range(len(X)):
#    if len(nltk.word_tokenize(row)) < minimum:
#        minimum = len(nltk.word_tokenize(row))
# print(minimum)

# Calculate the unique tokens by hand: --> Can't get this work: It only works with series type pandas data
# vector = {}
# X = pd.Series(X['essay_pre'])
# for i in range(len(X)):
#    for word in nltk.word_tokenize(X.iloc[i]):
#        if word not in vector.keys(X.iloc[i]):
#            vector[word] = 1

# the output looks to be a size 13650 words

# Delete the essays that have less than 200 words ~ approx. 20 essays

# Setting up the Keras LSTM Network
# the code below is taken from https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 975

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(X['essay_pre'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# there appears to be 11379 unique tokens

# Padding the sentences
X2 = tokenizer.texts_to_sequences(X['essay_pre'].values)
X2 = tf.keras.preprocessing.sequence.pad_sequences(X2, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X2.shape)

# Train/Test/Split:
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.10, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Building the model: --> Stuck on this
EMBEDDING_DIM = 100

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test, y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()
