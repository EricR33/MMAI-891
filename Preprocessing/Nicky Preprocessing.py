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
for document in file.essay:
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

###=======================================================
###the below is only for checking
###=======================================================

corpus5 = []

for paragraph in corpus2:
    document = []
    for sent in paragraph:
        sent = str(sent)
        sent = re.sub(r"@CAPS", " ", sent)
        sent = re.sub(r"@DATE", " ", sent)
        sent = re.sub(r"@LOCATION", " ", sent)
        sent = re.sub(r"@ORGANIZATION", " ", sent)
        sent = re.sub(r"@NUM", " ", sent)
        sent = re.sub(r"@PERCENT", " ", sent)
        sent = re.sub(r"@PERSON", " ", sent)
        sent = re.sub(r"@MONTH", " ", sent)
        sent = re.sub(r"@CITY", " ", sent)
        sent = re.sub(r"@YEAR", " ", sent)
        sent = re.sub(r"\d+", " ", sent)  ## remove digits
        sent = re.sub(r"\s+[a-z]\s+", " ", sent,
                      flags=re.I)  ## remove single character word such as "I", "a" in the middle of a sentence
        sent = re.sub(r"^[a-z]\s+", " ", sent,
                      flags=re.I)  ## remove single character word such as "I", "a" at the beginning of a sentence
        sent = re.sub(r"\s+[a-z]$", " ", sent,
                      flags=re.I)  ## remove single character word such as "I", "a" at the end of a sentence
        sent = re.sub(r"^\s", " ", sent)  ## remove space before sentence
        sent = re.sub(r"\s$", " ", sent)  ## remove space at the end of sentence
        sent = re.sub(r"[[0-9]]*]", " ", sent)  ## remove [9],[0],etc.
        sent = re.sub(r"\s+", " ", sent)  ## generated a lot of extra spaces so far so need to remove extra spaces
        sent = sent.lower()
        sent = re.sub(r"that's", "that is", sent)
        sent = re.sub(r"there's", "there is", sent)
        sent = re.sub(r"what's", "what is", sent)
        sent = re.sub(r"where's", "where is", sent)
        sent = re.sub(r"it's", "it is", sent)
        sent = re.sub(r"who's", "who is", sent)
        sent = re.sub(r"i'm", "i am", sent)
        sent = re.sub(r"she's", "she is", sent)
        sent = re.sub(r"he's", "he is", sent)
        sent = re.sub(r"they're", "they are", sent)
        sent = re.sub(r"who're", "who are", sent)
        sent = re.sub(r"you're", "you are", sent)
        sent = re.sub(r"ain't", "am not", sent)
        sent = re.sub(r"aren't", "are not", sent)
        sent = re.sub(r"wouldn't", "would not", sent)
        sent = re.sub(r"shouldn't", "should not", sent)
        sent = re.sub(r"couldn't", "could not", sent)
        sent = re.sub(r"doesn't", "does not", sent)
        sent = re.sub(r"isn't", "is not", sent)
        sent = re.sub(r"can't", "can not", sent)
        sent = re.sub(r"couldn't", "could not", sent)
        sent = re.sub(r"won't", "will not", sent)
        sent = re.sub(r"i've", "i have", sent)
        sent = re.sub(r"you've", "you have", sent)
        sent = re.sub(r"they've", "they have", sent)
        sent = re.sub(r"we've", "we have", sent)
        sent = re.sub(r"don't", "do not", sent)
        sent = re.sub(r"didn't", "did not", sent)
        sent = re.sub(r"i'll", "i will", sent)
        sent = re.sub(r"you'll", "you will", sent)
        sent = re.sub(r"he'll", "he will", sent)
        sent = re.sub(r"she'll", "she will", sent)
        sent = re.sub(r"they'll", "they will", sent)
        sent = re.sub(r"we'll", "we will", sent)
        sent = re.sub(r"i'd", "i would", sent)
        sent = re.sub(r"you'd", "you would", sent)
        sent = re.sub(r"he'd", "he would", sent)
        sent = re.sub(r"she'd", "she would", sent)
        sent = re.sub(r"they'd", "they would", sent)
        sent = re.sub(r"we'd", "we would", sent)
        sent = re.sub(r"she's", "she has", sent)
        sent = re.sub(r"he's", "he has", sent)
        document.append(sent)
    corpus5.append(document)

corpus5[0]
file1.essay[0]

###=======================================================
###the above is only for checking
###=======================================================







