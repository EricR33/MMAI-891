
# Import libraries
import numpy as np
import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
# from sklearn.datasets import load_files
nltk.download('stopwords')
# Load Data
df = pd.read_excel('training_set_rel3.xlsx', sheet_name='training_set')

# define function to preprocess

def prep_data(text):
    review = re.sub(r'\W', ' ', text)  # Remove all non words characters ( punctuation)
    review = review.lower()  # convert the review to a lower form
    review = re.sub(r'\s+[a-z]\s+', ' ', review)  # remove single character
    review = re.sub(r'^[a-z]\s+', ' ', review)  # remove single character at the begining
    # review = re.sub(r"\d"," ",review) # Remove different digits
    review = re.sub(r'\s+', ' ', review)  # Remove extra space
    # split into tokens by white space
    tokens = review.split()
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    return tokens

# Clean the the essay column
for i in range(0, df['essay'].shape[0]):
    df.at[i,'essay']= prep_data(df.at[i,'essay'])

# Save the dataframe to csv
df.to_csv("training_set_rel3.csv")

# Lemm

# Stem


#
