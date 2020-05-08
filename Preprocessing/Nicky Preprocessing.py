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
##file1 = pd.read_csv('C:/Users/Nicky/Desktop/Old_Com/Queens_Master/MMAI891/Project/training_set_rel4.txt', sep='\t', encoding = 'latin-1')


for i in range(len(file.essay)):
    file.essay[i] = str(file.essay[i])
    file.essay[i] = re.sub(r"@CAPS", " ", file.essay[i])
    file.essay[i] = re.sub(r"@DATE", " ", file.essay[i])
    file.essay[i] = re.sub(r"@LOCATION", " ", file.essay[i])
    file.essay[i] = re.sub(r"@ORGANIZATION", " ", file.essay[i])
    file.essay[i] = re.sub(r"@NUM", " ", file.essay[i])
    file.essay[i] = re.sub(r"@PERCENT", " ", file.essay[i])
    file.essay[i] = re.sub(r"@PERSON", " ", file.essay[i])
    file.essay[i] = re.sub(r"@MONTH", " ", file.essay[i])
    file.essay[i] = re.sub(r"@CITY", " ", file.essay[i])
    file.essay[i] = re.sub(r"@YEAR", " ", file.essay[i])
    file.essay[i] = re.sub(r"\W", " ", file.essay[i])  ## remove non word characters
    file.essay[i] = re.sub(r"\d", " ", file.essay[i])  ## remove digits
    file.essay[i] = re.sub(r"\s+[a-z]\s+", " ", file.essay[i],
                           flags=re.I)  ## remove single character word such as "I", "a" in the middle of a sentence
    file.essay[i] = re.sub(r"^[a-z]\s+", " ", file.essay[i],
                           flags=re.I)  ## remove single character word such as "I", "a" at the beginning of a sentence
    file.essay[i] = re.sub(r"^[a-z]\s+", " ", file.essay[i],
                           flags=re.I)  ## remove single character word such as "I", "a" at the end of a sentence
    file.essay[i] = re.sub(r"^s", " ", file.essay[i])  ## remove space before sentence
    file.essay[i] = re.sub(r"\s$", " ", file.essay[i])  ## remove space at the end of sentence
    file.essay[i] = re.sub(r"\[[0-9]]*\]", " ", file.essay[i])  ## remove [9],[0],etc.
    file.essay[i] = re.sub(r"\s+", " ",
                           file.essay[i])  ## generated a lot of extra spaces so far so need to remove extra spaces
    file.essay[i] = file.essay[i].lower()
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


def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems


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
    words = stem_words(words)
    words = lemmatize_verbs(words)
    document.append(words)
    corpus.append(document)

df = pd.DataFrame(corpus, columns=['essay_clean'])
file_clean = pd.concat([file, df], axis=1, sort=False)




