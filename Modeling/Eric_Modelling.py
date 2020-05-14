import pandas as pd
import os
import nltk
import re
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.preprocessing.text import Tokenizer
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import SpatialDropout1D, LSTM, Dense
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# pwd

# cd '/Users/ericross/School/Queens_MMAI/MMAI/MMAI_891/Project'

# Directories
DATASET_DIR = './asap-aes/'
WORKBOOK_DIR = './courses/'

# Import Data
df = pd.read_csv(os.path.join(DATASET_DIR, "training_set_rel3.tsv"), sep='\t', encoding='ISO-8859-1')

# Split X & Y
X = df['essay']
X = X.iloc[12253:]
print(X.iloc[0])
y = df['domain1_score']
y = y.iloc[12253:]
print(y.shape, X.shape)

# Regular Expression Cleaning
for i in range(len(X)):
    lemmatizer = WordNetLemmatizer()
    X.iloc[i] = X.iloc[i].lower()
    X.iloc[i] = re.sub(r'\.', '. ', X.iloc[i])
    X.iloc[i] = re.sub(r'\s+', ' ', X.iloc[i])
    # X.iloc[i] = re.sub(r"[abc\\]$", "'", X.iloc[i])        # trying to get rid of \'s notation
    X.iloc[i] = re.sub(r"@CAPS", " ", X.iloc[i])
    X.iloc[i] = re.sub(r"@DATE", " ", X.iloc[i])
    X.iloc[i] = re.sub(r"@LOCATION", " ", X.iloc[i])
    X.iloc[i] = re.sub(r"@ORGANIZATION", " ", X.iloc[i])
    X.iloc[i] = re.sub(r"@NUM", " ", X.iloc[i])
    X.iloc[i] = re.sub(r"@PERCENT", " ", X.iloc[i])
    X.iloc[i] = re.sub(r"@PERSON", " ", X.iloc[i])
    X.iloc[i] = re.sub(r"@MONTH", " ", X.iloc[i])
    X.iloc[i] = re.sub(r"@CITY", " ", X.iloc[i])
    X.iloc[i] = re.sub(r"@YEAR", " ", X.iloc[i])
    X.iloc[i] = re.sub(r'\d+', '', X.iloc[i])
    X.iloc[i] = re.sub(r"that's", "that is", X.iloc[i])
    X.iloc[i] = re.sub(r"there's", "there is", X.iloc[i])
    X.iloc[i] = re.sub(r"what's", "what is", X.iloc[i])
    X.iloc[i] = re.sub(r"where's", "where is", X.iloc[i])
    X.iloc[i] = re.sub(r"it's", "it is", X.iloc[i])
    X.iloc[i] = re.sub(r"who's", "who is", X.iloc[i])
    X.iloc[i] = re.sub(r"i'm", "i am", X.iloc[i])
    X.iloc[i] = re.sub(r"she's", "she is", X.iloc[i])
    X.iloc[i] = re.sub(r"he's", "he is", X.iloc[i])
    X.iloc[i] = re.sub(r"they're", "they are", X.iloc[i])
    X.iloc[i] = re.sub(r"who're", "who are", X.iloc[i])
    X.iloc[i] = re.sub(r"you're", "you are", X.iloc[i])
    X.iloc[i] = re.sub(r"ain't", "am not", X.iloc[i])
    X.iloc[i] = re.sub(r"aren't", "are not", X.iloc[i])
    X.iloc[i] = re.sub(r"wouldn't", "would not", X.iloc[i])
    X.iloc[i] = re.sub(r"shouldn't", "should not", X.iloc[i])
    X.iloc[i] = re.sub(r"couldn't", "could not", X.iloc[i])
    X.iloc[i] = re.sub(r"doesn't", "does not", X.iloc[i])
    X.iloc[i] = re.sub(r"isn't", "is not", X.iloc[i])
    X.iloc[i] = re.sub(r"can't", "can not", X.iloc[i])
    X.iloc[i] = re.sub(r"couldn't", "could not", X.iloc[i])
    X.iloc[i] = re.sub(r"won't", "will not", X.iloc[i])
    X.iloc[i] = re.sub(r"i've", "i have", X.iloc[i])
    X.iloc[i] = re.sub(r"you've", "you have", X.iloc[i])
    X.iloc[i] = re.sub(r"they've", "they have", X.iloc[i])
    X.iloc[i] = re.sub(r"we've", "we have", X.iloc[i])
    X.iloc[i] = re.sub(r"don't", "do not", X.iloc[i])
    X.iloc[i] = re.sub(r"didn't", "did not", X.iloc[i])
    X.iloc[i] = re.sub(r"i'll", "i will", X.iloc[i])
    X.iloc[i] = re.sub(r"you'll", "you will", X.iloc[i])
    X.iloc[i] = re.sub(r"he'll", "he will", X.iloc[i])
    X.iloc[i] = re.sub(r"she'll", "she will", X.iloc[i])
    X.iloc[i] = re.sub(r"they'll", "they will", X.iloc[i])
    X.iloc[i] = re.sub(r"we'll", "we will", X.iloc[i])
    X.iloc[i] = re.sub(r"i'd", "i would", X.iloc[i])
    X.iloc[i] = re.sub(r"you'd", "you would", X.iloc[i])
    X.iloc[i] = re.sub(r"he'd", "he would", X.iloc[i])
    X.iloc[i] = re.sub(r"she'd", "she would", X.iloc[i])
    X.iloc[i] = re.sub(r"they'd", "they would", X.iloc[i])
    X.iloc[i] = re.sub(r"we'd", "we would", X.iloc[i])
    X.iloc[i] = re.sub(r"she's", "she has", X.iloc[i])
    X.iloc[i] = re.sub(r"he's", "he has", X.iloc[i])

# Need to incorporate lemmatization into the For loop above during cleaning

# Finding the maximum words for an essay
maximum = 0
for i in range(len(X)):
    if len(nltk.word_tokenize(X.iloc[i])) > maximum:
        maximum = len(nltk.word_tokenize(X.iloc[i]))
print(maximum)

# Finding the minimum words for an essay
minimum = 1051
for i in range(len(X)):
    if len(nltk.word_tokenize(X.iloc[i])) < minimum:
        minimum = len(nltk.word_tokenize(X.iloc[i]))
print(minimum)

# Calculate the unique tokens by hand:
vector = {}
for i in range(len(X)):
    for word in nltk.word_tokenize(X.iloc[i]):
        if word not in vector.keys():
            vector[word] = 1

    # the output looks to be a size 13650 words

# Delete the essays that have less than 200 words ~ approx. 20 essays


# Will need to truncate / pad the sentences that are too short


# Setting up the Keras LSTM Network
# the code below is taken from https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
X = pd.DataFrame(data=X, columns=['essay'])

MAX_NB_WORDS = 5000
MAX_SEQUENCE_LENGTH = 975

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(X['essay'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# there appears to be 12185 unique tokens

# Padding the sentences --> Stuck on this
X2 = tokenizer.texts_to_sequences(X['essay'].values)
X2 = tf.keras.preprocessing.sequence.pad_sequences(X2, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X2.shape)

# Train/Test/Split:
X_train, X_test, Y_train, Y_test = train_test_split(X2, y, test_size=0.10, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# Building the model:
EMBEDDING_DIM = 100

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test, Y_test)
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
