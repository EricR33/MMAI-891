# -*- coding: utf-8 -*-
"""Modelling_NLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15gpf59pfXNYbtxnLYQS3Y5t5HRZ6t96g
"""

pip
install
textstat

!pip
install
pyyaml
h5py

import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import textstat
import collections
from collections import defaultdict
import os
from sklearn import preprocessing
from keras.preprocessing import sequence
##
##pip install https://github.com/dmlc/gluon-nlp/tarball/master
##from bert-embedding import BertEmbedding

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
import random
import math

from google.colab import drive

drive.mount('/content/gdrive')

DATASET_DIR = 'gdrive/My Drive/MMAI891/'

##DATASET_DIR = '/content/drive/My Drive/MMAI891/training_set_rel4.txt'


file = pd.read_csv(os.path.join(DATASET_DIR, "new_df-5.csv"))

corpus = file['Essay_Prep']
features = file.iloc[:, 4:]

max(corpus.apply(len))

from keras.preprocessing.text import Tokenizer

max_words = 9000  # to define vocab size for max num of words to keep based on word freq, here we are only keeping the 6000-1 most common words
##max_len = 150 #to define fixed sequence length, here we are padding the input sequence to have the same length of 150

# vectorize the corpus
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(
    corpus)  # Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency
sequences = tok.texts_to_sequences(corpus)  # Transforms each text in texts to a sequence of integers.
sequences_matrix = sequence.pad_sequences(sequences)  # pad the vector so they are all the same length

word_index = tok.word_index

# Load pre-trained embedding matrix from Glove

embeddings_index = dict()
f = open(os.path.join(DATASET_DIR, 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Loaded %s word vectors.' % len(embeddings_index))

# Prepare embedding vectors for our data

embedding_matrix = np.zeros((len(word_index) + 1, 100), dtype='float32')

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)  # look up the word in the pre-built embedding matrix

    # words not found will have a vector of all-zeros, otherwise, return the embedding vector
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

standardized_features = preprocessing.scale(features)
standardized_features.shape
features_list = standardized_features.tolist()
sequences_matrix_list = sequences_matrix.tolist()

document_with_tar = [[document, file['Essay Score'][i]] for i, document in enumerate(sequences_matrix_list)]
combine = [[j[-1]] + i + j[:-1] for i, j in zip(features_list, document_with_tar)]
combine[1][14]
len(combine[1])
random.shuffle(combine)
document_labels = [(feature[1:14] + [feature[14]][0], feature[0]) for feature in combine]
document_labels[0]

processed_dataset = tf.data.Dataset.from_generator(lambda: document_labels, output_types=(tf.float32, tf.int64))
BATCH_SIZE = 32
# batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
batched_dataset = processed_dataset.batch(BATCH_SIZE)
next(iter(batched_dataset))

TOTAL_BATCHES = math.ceil(len(document_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 5
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

next(iter(train_data))

checkpoint_path = os.path.join(DATASET_DIR, "training_1/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=100,
                 embedding_matrix=embedding_matrix,
                 max_len=861,
                 dnn_units=256,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size, embedding_dimensions, weights=[embedding_matrix],
                                          # using the pre-trained embedding
                                          input_length=max_len,
                                          trainable=False)
        self.rnn_layer1 = layers.LSTM(256)  ##, input_shape=(None, 1))

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="linear")

    def call(self, inputs, training):
        nn_input = tf.slice(inputs, [0, 13], [-1, -1])
        l = self.embedding(nn_input)
        ##nn_input = tf.expand_dims(nn_input, -1)
        l_1 = self.rnn_layer1(l)

        feature_input = tf.slice(inputs, [0, 0], [-1, 13])
        feature_input = tf.cast(feature_input, dtype=tf.float32)
        concatenated = tf.concat([l_1, feature_input], axis=-1)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(word_index) + 1
EMB_DIM = 100
DNN_UNITS = 256
DROPOUT_RATE = 0.2
NB_EPOCHS = 20
max_len = 861

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        embedding_matrix=embedding_matrix,
                        max_len=max_len,
                        dnn_units=DNN_UNITS,
                        dropout_rate=DROPOUT_RATE)

text_model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['mean_absolute_error'])

# Commented out IPython magic to ensure Python compatibility.
# %%time
# text_model.fit(train_data, epochs=NB_EPOCHS, callbacks=[cp_callback])

!ls
{checkpoint_dir}

text_model.load_weights(checkpoint_path)

results = text_model.evaluate(test_data)
print(results)

checkpoint_path = os.path.join(DATASET_DIR, "training_2/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=100,
                 embedding_matrix=embedding_matrix,
                 dnn_units=256,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size, embedding_dimensions, weights=[embedding_matrix],
                                          # using the pre-trained embedding
                                          ##input_length=max_len,
                                          trainable=False)
        self.rnn_layer1 = layers.LSTM(256)  ##, input_shape=(None, 1))

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="linear")

    def call(self, inputs, training):
        nn_input = tf.slice(inputs, [0, 13], [-1, -1])
        l = self.embedding(nn_input)
        ##nn_input = tf.expand_dims(nn_input, -1)
        l_1 = self.rnn_layer1(l)

        feature_input = tf.slice(inputs, [0, 0], [-1, 13])
        feature_input = tf.cast(feature_input, dtype=tf.float32)
        concatenated = tf.concat([l_1, feature_input], axis=-1)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(word_index) + 1
EMB_DIM = 100
DNN_UNITS = 256
DROPOUT_RATE = 0.2
NB_EPOCHS = 20
max_len = 861

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        embedding_matrix=embedding_matrix,
                        ##max_len = max_len,
                        dnn_units=DNN_UNITS,
                        dropout_rate=DROPOUT_RATE)

text_model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['mean_absolute_error'])

# Commented out IPython magic to ensure Python compatibility.
# %%time
# text_model.fit(train_data, epochs=NB_EPOCHS, callbacks=[cp_callback])

!ls
{checkpoint_dir}

results = text_model.evaluate(test_data)
print(results)

text_model.load_weights(checkpoint_path)

results = text_model.evaluate(test_data)
print(results)

checkpoint_path = os.path.join(DATASET_DIR, "training_3/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


###==========================
## With no manual features
##==========================

class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=100,
                 embedding_matrix=embedding_matrix,
                 dnn_units=256,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size, embedding_dimensions, weights=[embedding_matrix],
                                          # using the pre-trained embedding
                                          ##input_length=max_len,
                                          trainable=False)
        self.rnn_layer1 = layers.LSTM(256)

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="linear")

    def call(self, inputs, training):
        nn_input = tf.slice(inputs, [0, 13], [-1, -1])
        l = self.embedding(nn_input)
        l_1 = self.rnn_layer1(l)

        ##feature_input = tf.slice(inputs, [0, 0], [-1, 4])
        ##feature_input = tf.cast(feature_input, dtype = tf.float32)
        ##concatenated = tf.concat([l_1, feature_input], axis=-1)
        concatenated = self.dense_1(l_1)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(word_index) + 1
EMB_DIM = 100
DNN_UNITS = 256
DROPOUT_RATE = 0.2
NB_EPOCHS = 20

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        embedding_matrix=embedding_matrix,
                        dnn_units=DNN_UNITS,
                        dropout_rate=DROPOUT_RATE)

text_model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['mean_absolute_error'])

# Commented out IPython magic to ensure Python compatibility.
# %%time
# text_model.fit(train_data, epochs=NB_EPOCHS, callbacks=[cp_callback])

!ls
{checkpoint_dir}

results = text_model.evaluate(test_data)
print(results)

text_model.load_weights(checkpoint_path)

results = text_model.evaluate(test_data)
print(results)


###==========================
## manual features attached to the input feeding into NN
##==========================

class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=100,
                 embedding_matrix=embedding_matrix,
                 dnn_units=256,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size, embedding_dimensions, weights=[embedding_matrix],
                                          # using the pre-trained embedding
                                          input_length=max_len,
                                          trainable=False)
        self.rnn_layer1 = layers.LSTM(256)

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="linear")

    def call(self, inputs, training):
        ##nn_input = tf.slice(inputs, [0, 4], [-1, -1])
        l = self.embedding(inputs)
        l_1 = self.rnn_layer1(l)

        ##feature_input = tf.slice(inputs, [0, 0], [-1, 4])
        ##feature_input = tf.cast(feature_input, dtype = tf.float32)
        ##concatenated = tf.concat([l_1, feature_input], axis=-1)
        concatenated = self.dense_1(l_1)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(word_index) + 1
EMB_DIM = 100
DNN_UNITS = 256
DROPOUT_RATE = 0.2
NB_EPOCHS = 20

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        embedding_matrix=embedding_matrix,
                        dnn_units=DNN_UNITS,
                        dropout_rate=DROPOUT_RATE)

text_model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['mean_absolute_error'])

text_model.fit(train_data, epochs=NB_EPOCHS)

###==========================
## Attention
##==========================


checkpoint_path = os.path.join(DATASET_DIR, "training_4/cp.ckpt")
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


class Attention(tf.keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = tf.nn.tanh(
            self.W1(features) + self.W2(hidden_with_time_axis))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=100,
                 embedding_matrix=embedding_matrix,
                 dnn_units=256,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size, embedding_dimensions, weights=[embedding_matrix],
                                          # using the pre-trained embedding
                                          ##input_length=max_len,
                                          trainable=False)

        ##self.rnn_layer1 = layers.LSTM(256) ##, input_shape=(None, 1))
        self.rnn_layer2 = layers.Bidirectional(layers.LSTM(32, return_sequences=True), name="bi_lstm_0")
        self.rnn_layer3 = layers.Bidirectional(layers.LSTM(32, return_sequences=True, return_state=True),
                                               name="bi_lstm_1")
        self.concat = layers.Concatenate(axis=-1)
        self.attention = Attention(10)

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="linear")

    def call(self, inputs, training):
        nn_input = tf.slice(inputs, [0, 13], [-1, -1])
        l = self.embedding(nn_input)
        ##nn_input = tf.expand_dims(nn_input, -1)
        l_1 = self.rnn_layer2(l)
        (lstm, forward_h, forward_c, backward_h, backward_c) = self.rnn_layer3(l_1)
        state_h = self.concat([forward_h, backward_h])
        state_c = self.concat([forward_c, backward_c])
        context_vector, attention_weights = self.attention(lstm, state_h)

        feature_input = tf.slice(inputs, [0, 0], [-1, 13])
        feature_input = tf.cast(feature_input, dtype=tf.float32)
        concatenated = tf.concat([context_vector, feature_input], axis=-1)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(word_index) + 1
EMB_DIM = 100
DNN_UNITS = 256
DROPOUT_RATE = 0.2
NB_EPOCHS = 20
max_len = 861

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        embedding_matrix=embedding_matrix,
                        ##max_len = max_len,
                        dnn_units=DNN_UNITS,
                        dropout_rate=DROPOUT_RATE)

text_model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['mean_absolute_error'])

# Commented out IPython magic to ensure Python compatibility.
# %%time
# text_model.fit(train_data, epochs=NB_EPOCHS, callbacks=[cp_callback])

!ls
{checkpoint_dir}

results = text_model.evaluate(test_data)
print(results)

text_model.load_weights(checkpoint_path)

results = text_model.evaluate(test_data)
print(results)