# -*- coding: utf-8 -*-
"""
Created on Wed May  6 22:12:31 2020

@author: Nicky
"""

import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

file = pd.read_csv('C:/Users/Nicky/Desktop/Old_Com/Queens_Master/MMAI891/Project/training_set_rel4.txt', sep='\t',
                   encoding='latin-1')
file1 = pd.read_csv('C:/Users/Nicky/Desktop/Old_Com/Queens_Master/MMAI891/Project/training_set_rel4.txt', sep='\t',
                    encoding='latin-1')

for i in range(len(file.essay)):
    file.essay[i] = str(file.essay[i])
    file.essay[i] = file.essay[i].lower()
    file.essay[i] = re.sub(r"@caps", " ", file.essay[i])
    file.essay[i] = re.sub(r"@date", " ", file.essay[i])
    file.essay[i] = re.sub(r"@location", " ", file.essay[i])
    file.essay[i] = re.sub(r"@organization", " ", file.essay[i])
    file.essay[i] = re.sub(r"@num", " ", file.essay[i])
    file.essay[i] = re.sub(r"@percent", " ", file.essay[i])
    file.essay[i] = re.sub(r"@person", " ", file.essay[i])
    file.essay[i] = re.sub(r"@month", " ", file.essay[i])
    file.essay[i] = re.sub(r"@city", " ", file.essay[i])
    file.essay[i] = re.sub(r"@year", " ", file.essay[i])
    file.essay[i] = re.sub(r"dear editor", " ", file.essay[i])
    file.essay[i] = re.sub(r"dear newspaper", " ", file.essay[i])
    file.essay[i] = re.sub(r"dear local newspaper", " ", file.essay[i])
    file.essay[i] = re.sub(r"newspaper", " ", file.essay[i])
    file.essay[i] = re.sub(r"that's", "that is", file.essay[i])
    file.essay[i] = re.sub(r"there's", "there is", file.essay[i])
    file.essay[i] = re.sub(r"what's", "what is", file.essay[i])
    file.essay[i] = re.sub(r"where's", "where is", file.essay[i])
    file.essay[i] = re.sub(r"it's", "it is", file.essay[i])
    file.essay[i] = re.sub(r"who's", "who is", file.essay[i])
    file.essay[i] = re.sub(r"i'm", "i am", file.essay[i])
    file.essay[i] = re.sub(r"she's", "she is", file.essay[i])
    file.essay[i] = re.sub(r"he's", "he is", file.essay[i])
    file.essay[i] = re.sub(r"they're", "they are", file.essay[i])
    file.essay[i] = re.sub(r"who're", "who are", file.essay[i])
    file.essay[i] = re.sub(r"you're", "you are", file.essay[i])
    file.essay[i] = re.sub(r"ain't", "am not", file.essay[i])
    file.essay[i] = re.sub(r"aren't", "are not", file.essay[i])
    file.essay[i] = re.sub(r"wouldn't", "would not", file.essay[i])
    file.essay[i] = re.sub(r"shouldn't", "should not", file.essay[i])
    file.essay[i] = re.sub(r"couldn't", "could not", file.essay[i])
    file.essay[i] = re.sub(r"doesn't", "does not", file.essay[i])
    file.essay[i] = re.sub(r"isn't", "is not", file.essay[i])
    file.essay[i] = re.sub(r"can't", "can not", file.essay[i])
    file.essay[i] = re.sub(r"couldn't", "could not", file.essay[i])
    file.essay[i] = re.sub(r"won't", "will not", file.essay[i])
    file.essay[i] = re.sub(r"i've", "i have", file.essay[i])
    file.essay[i] = re.sub(r"you've", "you have", file.essay[i])
    file.essay[i] = re.sub(r"they've", "they have", file.essay[i])
    file.essay[i] = re.sub(r"we've", "we have", file.essay[i])
    file.essay[i] = re.sub(r"don't", "do not", file.essay[i])
    file.essay[i] = re.sub(r"didn't", "did not", file.essay[i])
    file.essay[i] = re.sub(r"i'll", "i will", file.essay[i])
    file.essay[i] = re.sub(r"you'll", "you will", file.essay[i])
    file.essay[i] = re.sub(r"he'll", "he will", file.essay[i])
    file.essay[i] = re.sub(r"she'll", "she will", file.essay[i])
    file.essay[i] = re.sub(r"they'll", "they will", file.essay[i])
    file.essay[i] = re.sub(r"we'll", "we will", file.essay[i])
    file.essay[i] = re.sub(r"i'd", "i would", file.essay[i])
    file.essay[i] = re.sub(r"you'd", "you would", file.essay[i])
    file.essay[i] = re.sub(r"he'd", "he would", file.essay[i])
    file.essay[i] = re.sub(r"she'd", "she would", file.essay[i])
    file.essay[i] = re.sub(r"they'd", "they would", file.essay[i])
    file.essay[i] = re.sub(r"we'd", "we would", file.essay[i])
    file.essay[i] = re.sub(r"she's", "she has", file.essay[i])
    file.essay[i] = re.sub(r"he's", "he has", file.essay[i])
    file.essay[i] = re.sub(r"\W", " ", file.essay[i])  ## remove non word characters
    file.essay[i] = re.sub(r"\d+", " ", file.essay[i])  ## remove digits
    file.essay[i] = re.sub(r"\s+[a-z]\s+", " ", file.essay[i],
                           flags=re.I)  ## remove single character word such as "I", "a" in the middle of a sentence
    file.essay[i] = re.sub(r"^[a-z]\s+", " ", file.essay[i],
                           flags=re.I)  ## remove single character word such as "I", "a" at the beginning of a sentence
    file.essay[i] = re.sub(r"\s+[a-z]$", " ", file.essay[i],
                           flags=re.I)  ## remove single character word such as "I", "a" at the end of a sentence
    file.essay[i] = re.sub(r"^\s+", " ", file.essay[i])  ## remove space before sentence
    file.essay[i] = re.sub(r"\s+$", " ", file.essay[i])  ## remove space at the end of sentence
    file.essay[i] = re.sub(r"\[[0-9]]*\]", " ", file.essay[i])  ## remove [9],[0],etc.
    file.essay[i] = re.sub(r"\s+", " ",
                           file.essay[i])  ## generated a lot of extra spaces so far so need to remove extra spaces


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


##def stem_words(words):
##"""Stem words in list of tokenized words"""
##stemmer = LancasterStemmer()
##stems = []
##for word in words:
##stem = stemmer.stem(word)
##stems.append(stem)
##return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


corpus = []

for i in range(len(file.essay)):
    document = []
    words = nltk.word_tokenize(file.essay[i])
    words = remove_stopwords(words)
    ##words = stem_words(words)
    words = lemmatize_verbs(words)
    document.append(words)
    corpus.append(document)

df = pd.DataFrame(corpus, columns=['essay_clean'])
file_clean = pd.concat([file, df], axis=1, sort=False)

corpus[0]

## the below is preprocessing step for BERT model
## https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
##pip install pytorch-pretrained-bert

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

try:
    % tensorflow_version
    2.
    x
except Exception:
    pass
import tensorflow as tf

##pip install tensorflow_hub
import tensorflow_hub as hub

from tensorflow.keras import layers
import random
import math

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

nltk.download('punkt')

corpus1 = []
for i in range(len(file1.essay)):
    document = []
    sents = nltk.tokenize.sent_tokenize(file1.essay[i])
    document.append(sents)
    corpus1.append(document)
corpus1[0][0]

##corpus2 = []
##for paragraph in corpus1:
##document = []
##for sent in paragraph[0]:
##marked_sent = "[CLS] " + str(sent) + " [SEP]"
##document.append(marked_sent)
##corpus2.append(document)

file1.essay[0]

corpus3 = []
for document in file[file.essay_set == 1].essay:
    tokenized_text = tokenizer.tokenize(document)
    if len(tokenized_text) > 512:
        tokenized_text = tokenized_text[:512]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    corpus3.append(indexed_tokens)

###==========================================
###method according to https://stackabuse.com/text-classification-with-bert-tokenizer-and-tf-2-0-in-python/
###============================================


document_with_len = [[document, file1.domain1_score[i], len(document)] for i, document in enumerate(corpus3)]
random.shuffle(document_with_len)
document_with_len.sort(key=lambda x: x[2])
sorted_document_labels = [(feature[0], feature[1]) for feature in document_with_len]
processed_dataset = tf.data.Dataset.from_generator(lambda: sorted_document_labels, output_types=(tf.int64, tf.int64))
BATCH_SIZE = 32
batched_dataset = processed_dataset.padded_batch(BATCH_SIZE, padded_shapes=((None,), ()))
next(iter(batched_dataset))

TOTAL_BATCHES = math.ceil(len(sorted_document_labels) / BATCH_SIZE)
TEST_BATCHES = TOTAL_BATCHES // 5
batched_dataset.shuffle(TOTAL_BATCHES)
test_data = batched_dataset.take(TEST_BATCHES)
train_data = batched_dataset.skip(TEST_BATCHES)

next(iter(train_data))

len(set(file[file.essay_set == 1].domain1_score))


###==========================================
###classfication model
###============================================


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 dnn_units=256,
                 model_output_classes=11,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.rnn_layer1 = layers.LSTM(256)  ###, input_shape=(218,1), return_sequences=False)

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")
        ###if want to do regression: self.last_dense = layers.Dense(units=1, activation="linear")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.rnn_layer1(l)

        ##concatenated = tf.concat([l_1, l_2, l_3], axis=-1)
        concatenated = self.dense_1(l_1)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 128
DNN_UNITS = 64
OUTPUT_CLASSES = 13
DROPOUT_RATE = 0.2
NB_EPOCHS = 5

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)

if OUTPUT_CLASSES == 2:
    text_model.compile(loss="binary_crossentropy",
                       optimizer="adam",
                       metrics=["accuracy"])
else:
    text_model.compile(loss="sparse_categorical_crossentropy",
                       optimizer="adam",
                       metrics=["sparse_categorical_accuracy"])
    ###text_model.compile(loss="mean_absolute_percentage_error", optimizer="adam")

text_model.fit(train_data, epochs=NB_EPOCHS)
results = text_model.evaluate(test_data)
print(results)


###==========================================
###regression method
###============================================


class TEXT_MODEL(tf.keras.Model):

    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 dnn_units=256,
                 dropout_rate=0.1,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)

        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.rnn_layer1 = layers.LSTM(256)

        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.last_dense = layers.Dense(units=1, activation="linear")

    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.rnn_layer1(l)

        ##concatenated = tf.concat([l_1, l_2, l_3], axis=-1)
        concatenated = self.dense_1(l_1)
        concatenated = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated)

        return model_output


VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 128
DNN_UNITS = 256
DROPOUT_RATE = 0.2
NB_EPOCHS = 5

text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        dnn_units=DNN_UNITS,
                        dropout_rate=DROPOUT_RATE)

text_model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['mean_absolute_error'])

##from keras.callbacks import ModelCheckpoint

##checkpoint_name = 'Weights-{epoch:03d}--{val_accuracy:.5f}.hdf5'
##checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
##callbacks_list = [checkpoint]

text_model.fit(train_data, epochs=NB_EPOCHS)
results = text_model.evaluate(test_data)
print(results)

