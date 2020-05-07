import nltk
import numpy as np
import heapq
import re

paragraph = 'I saw a dog'

dataset = nltk.sent_tokenize(paragraph)             # tokenizes the dataset into sentences

# Cleaning
for i in range(len(dataset)):
    dataset[i] = dataset[i].lower()                 # lowers all ASCII characters
    dataset[i] = re.sub(r'\W', ' ', dataset[i])     # Cleans out any non-ASCII characters
    dataset[i] = re.sub(r'\s+', ' ', dataset[i])    # Converts any whitespace 1 or more into 1 space


# Creating Histogram
word2count = {}
for data in dataset:
    words = nltk.word_tokenize(data)                # tokenizes the sentences into word based tokens
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1                    # creates a 1 value in the dictionary if the word doesn't exist
        else:
            word2count[word] += 1                   # adds a +1 value to the dictionary if the word does exist already

freq_words = heapq.nlargest(100, word2count, key=word2count.get)

# Creatinig the bag of words array
x = []
for data in dataset:
    vector = []
    for word in freq_words:
        if word in nltk.word_tokenize(data):
            vector.append(1)                    # appends a 1 to the vector list if the frequent word is in the tokenized sentence
        else:
            vector.append(0)                    # appends a 0 to the vector list if the frequent word is not in the tokenized sentence
    x.append(vector)                            # appends the vector list to the x list which has a shape of 20 (documents) x 100 (frequent words

x = np.asarray(x)