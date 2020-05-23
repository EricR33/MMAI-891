import os
# import nltk
import re, collections
import unidecode
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import SpatialDropout1D, LSTM, Dense, Embedding
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from Modeling.final_preprocessing import lem, pre_process, count_spell_error

# pwd
os.getcwd()

os.chdir('/Users/ericross/School/Queens_MMAI/MMAI/MMAI_891/Project')

# Directories
DATASET_DIR = './asap-aes/'
WORKBOOK_DIR = './courses/'

# Import Data For Mispelled Word Counter
df1 = pd.read_csv(os.path.join(DATASET_DIR, "training_set_rel3.tsv"), sep='\t', encoding='ISO-8859-1')
df1 = df1[12253:].copy()
df1 = pd.DataFrame(data=df1, columns=["essay","domain1_score"])

# Import Data from Spell Checked Document
df = pd.read_csv(os.path.join(DATASET_DIR, "Essays_SpellCheck_Set8.csv"))
df = df.drop(['Unnamed: 0'], axis=1)
df.info()
df.head()

# stop_words = set(stopwords.words('english') + stopwords.word('french'))   # haven't used this below

# Delete the essays that have less than 200 words ~ approx. 20 essays

# Apply preprocessing onto both dataframes
df['Essay_Prep'] = df['Essay'].apply(pre_process)
df1['Essay_Prep'] = df1['essay'].apply(pre_process)

# Check for Mispelled Words in Original DataFrame (df1)
# df1['Spelling_Mistakes_Count'] = df1['Essay_Prep'].apply(count_spell_error)

print(df.shape, df1.shape)

# Split X & Y
X = df['Essay_Prep'].values
y = df['Essay Score'].values

print(y.shape, type(y), X.shape, type(X))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_dict = tokenizer.index_word

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index
print("Found {0} unique words: ".format(len(word_index)))

# Padding
X_train_pad = pad_sequences(X_train_seq, maxlen=500, padding='post')    # change the maxlen
X_test_pad = pad_sequences(X_test_seq, maxlen=500, padding='post')      # change the maxlen
X_train_pad[:5]

print(X_train_pad.shape)

# LSTM Network
MAX_NB_WORDS = 9799
EMBEDDING_DIM = 100

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=MAX_NB_WORDS+1, output_dim=500, input_length=X_train_pad.shape[1]))
lstm_model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2))

lstm_model.add(Dense(1, activation='linear'))

lstm_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
lstm_model.summary()

# Setting up the Keras LSTM Network
# the code below is taken from https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
# Building the model: --> Stuck on this
# model.add(SpatialDropout1D(0.2))
# model.add(Dense(13, activation='linear'))

epochs = 3
batch_size = 64

# Having issues with this line of code --> i need to figure out how to run 1 essay at a time instead of the whole corpus
history = lstm_model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

import matplotlib.pyplot as plt
training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss) + 1)

plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot the accuracy
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot((history.history['mean_squared_error']), 'r', label='train')
ax.plot((history.history['val_mean_squared_error']), 'b', label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Mean Squared Error', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)



# accr = model.evaluate(X_test, y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

