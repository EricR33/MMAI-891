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

# Creating the bag of words array
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
        print(f'the frequent word is: {word}, w is: {w},the frequency is: {frequency}, '
              f'the document length is: {len(nltk.word_tokenize(data))},'
              f'the tf_word calculation is: {tf_word}')
        doc_tf.append(tf_word)
    tf_matrix[word] = doc_tf

# IDF Calculation
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
for word in tf_matrix.keys():
    tfidf = []
    for value in tf_matrix[word]:
        score = value * word_idfs[word]
        tfidf.append(score)