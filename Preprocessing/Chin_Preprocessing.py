import inline as inline
#%matplotlib inline

import numpy as np
import pandas as pd
import re
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-colorblind')

# Setup Pandas
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
pd.set_option('display.max_colwidth', 100)

plt.rcParams['figure.dpi']= 100

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import re, string, unicodedata
import nltk
#`nltk.download('popular')
#import contractions
#import inflect
#from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

import pprint
import pickle
# From SymSpellPy
import pkg_resources
from symspellpy.symspellpy import SymSpell, Verbosity
#######################################################


df = pd.read_csv("training_set_rel3.tsv", sep='\t', encoding='ISO-8859-1')
df = df.dropna(axis=1)
columns = ['rater1_domain1', 'rater2_domain1']
df = df.drop(columns,axis=1)
print(df.head())

df.info()

df.groupby('essay_set').agg('count').plot.bar(y='essay', rot=0, legend=False)
plt.title('Essay count by set #')
plt.ylabel('Count')
plt.show()

print(df.groupby('essay_set').agg('count'))

# Count characters and words for each essay
df['word_count'] = df['essay'].str.strip().str.split().str.len()

#Visualize characters and words for each essay
df.hist(column='word_count', by='essay_set', bins=25, sharey=True, sharex=True, layout=(2, 4), figsize=(7,4), rot=0)
plt.suptitle('Word count by topic #')
plt.xlabel('Number of words')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#Pick essay from row 1870 and correct spelling/grammar errors using textblob
text = df.essay[1783]
data = TextBlob(text)
print (data.correct())

#Select all the essays from set 1
set1 = df.loc[df['essay_set']==1]
print(set1)
texts = set1['essay']
scores = set1['domain1_score']

print(texts)

# SymSpellPy

# maximum edit distance per dictionary precalculation
max_edit_distance_dictionary = 2
prefix_length = 7


def findSuggestions(max_edit_distance_dictionary, prefix_length):
    # create object
    sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)

    # load dictionary
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

    # term_index is the column of the term and count_index is the
    # column of the term frequency
    if not sym_spell.load_dictionary(dictionary_path, term_index=0,
                                     count_index=1):
        print("Dictionary file not found")
        return

    print("Dictionary found")

    if not sym_spell.load_bigram_dictionary(bigram_path, term_index=0,
                                            count_index=2):
        print("Bigram dictionary file not found")
        return

    print("Bigram dictionary found")

    # max edit distance per lookup (per single word, not per whole input string)
    max_edit_distance_lookup = 2

    suggestions = [sym_spell.lookup_compound(input_term,
                                             max_edit_distance_lookup,
                                             transfer_casing=True)
                   for input_term in texts]
    return suggestions

suggest = findSuggestions(max_edit_distance_dictionary, prefix_length)

print (suggest)

# Create a new table to combine all the elements
result = []

for i in np.arange(0, len(texts)):
    elm=[suggest[i][0].term, scores[i]]
    result.append(elm)

pprint.pprint([result[i] for i in np.arange(1, 3)])

# Create Pickle File

file = open('Essay1_SpellCheck.p',"wb")

pickle.dump(result,file)

file.close()

## Load the data from the pickle file

data_set = pickle.load(open("Essay1_SpellCheck.p", "rb"))

## Converted to np_array for conversion
data_nparray = np.asarray(data_set)

## Save as a csv, without headers
headers=["Sr.No", "Essay", "Essay Score"]
pd.DataFrame(data_nparray).to_csv("Essay_SpellCheck.csv", header=headers)