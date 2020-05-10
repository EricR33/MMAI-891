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

# TF Calculation
tf_matrix = {}
for word in freq_words:     # iterates through the frequent words list
    doc_tf = []             # create an empty list for doc_tf
    for data in dataset:    # breaks down the dataset into individual documents (in this individual sentences)
        frequency = 0           # set the frequency of the term to 0
        for w in nltk.word_tokenize(data):  # tokenizes the document into individual words
            if w == word:                       # if the tokenized word is equal to the frequent word
                frequency += 1                  # add 1 to the frequency
        tf_word = frequency / len(nltk.word_tokenize(data))
        doc_tf.append(tf_word)
    tf_matrix[word] = doc_tf

#IDF Calculation
word_idfs = {}
for word in freq_words:     # iterates through the frequent words list
    doc_count = 0           # sets the document counter to 0
    for data in dataset:    # breaks down the dataset into documents --> in this example: breaks down paragraph into a sentences string
        if word in nltk.word_tokenize(data):    # if the frequent word is inside the sentences tokenized word list
            doc_count += 1                      # return + 1

    word_idfs[word] = np.log(len(dataset)/doc_count)    # log(divides the dataset length by the document count)

    # for the last case --> there are 22 documents (sentences) in the dataset and lastly appears 1 time
    # in the entire dataset, which equals = log(21 / 1) = 3.0445
    # for real world libraries --> np.log((len(dataset)/doc_count) + 1) --> the +1 is a bias

# TF - IDF Calculation
tfidf_matrix = []
for word in tf_matrix.keys():               # iterates through the term frequency keys
    tfidf = []                              # creates a blank list
    for value in tf_matrix[word]:           # iterates through the tf_matrix dictionary values based on the term frequency key slected
        score = value * word_idfs[word]     # Multiplies the TF values by the IDF value for the word
        print(f'the word key is: {word}, the value is: {value}, the score is {score},\n the idf word is {word_idfs[word]} ')
        tfidf.append(score)                 # Append the score to the tfidf list which contains the list for 1 key
    tfidf_matrix.append(tfidf)              # Append all of the scores relating to the keys back into the large matrix