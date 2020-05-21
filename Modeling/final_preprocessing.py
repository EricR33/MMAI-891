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
from Modeling.final_preprocessing import preprocess

# Regular Expression Cleaning
def print_string(x):
    return x



# Lemmatizer needs to be called after we check for spelling mistakes


def lem(x):
    lemmer = WordNetLemmatizer()
    x = [lemmer.lemmatize(w) for w in x.split()]
    return ' '.join(x)

# checking number of misspelled words (This is taken from https://github.com/shubhpawar/Automated-Essay-Scoring)


def count_spell_error(essay):
        # big.txt: It is a concatenation of public domain book excerpts from Project Gutenberg
    #         and lists of most frequent words from Wiktionary and the British National Corpus.
    #         It contains about a million words.
    data = open('big.txt').read()

    words_ = re.findall('[a-z]+', data.lower())

    word_dict = collections.defaultdict(lambda: 0)

    for word in words_:
        word_dict[word] += 1

    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

    mispell_count = 0

    words = clean_essay.split()

    for word in words:
        if not word in word_dict:
            mispell_count += 1

    return mispell_count