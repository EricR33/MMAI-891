import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import SpatialDropout1D, LSTM, Dense, Embedding
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# pwd
os.getcwd()

os.chdir('/Users/ericross/School/Queens_MMAI/MMAI/PyCharm Projects/MMAI-891/Modeling')



# Import Data For Mispelled Word Counter
data = pd.read_csv("new_df-5.csv")

df = data.drop(['Unnamed: 0'], axis=1)
df.info()
df.head()

print(df.shape)

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

epochs = 4
batch_size = 64

history = lstm_model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss) + 1)

plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

