import os
# import nltk
import re, collections
import unidecode
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import SpatialDropout1D, LSTM, Dense, Embedding
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# pwd

os.chdir('/Users/ericross/School/Queens_MMAI/MMAI/MMAI_891/Project')

# Directories
DATASET_DIR = './asap-aes/'
WORKBOOK_DIR = './courses/'

# Import Data For Mispelled Word Counter
df1 = pd.read_csv(os.path.join(DATASET_DIR, "training_set_rel3.tsv"), sep='\t', encoding='ISO-8859-1')
df1 = df1[12253:].copy()
df1 = pd.DataFrame(data=df1, columns=["essay","domain1_score"])

# Import Data from Spell Checked Document
df = pd.read_csv(os.path.join(DATASET_DIR, "Essays_SpellCheck_Set8.csv"))
df = df.drop(['Unnamed: 0'], axis=1)
df.info()
df.head()

# stop_words = set(stopwords.words('english') + stopwords.word('french'))   # haven't used this below

lemmer = WordNetLemmatizer()

# Regular Expression Cleaning


def preprocess(x):
    # x = re.sub(r"[abc\\]$", "'", x)        # trying to get rid of \'s notation
    x = re.sub(r"@CAPS", " ", x)
    x = re.sub(r"@DATE", " ", x)
    x = re.sub(r"@LOCATION", " ", x)
    x = re.sub(r"@ORGANIZATION", " ", x)
    x = re.sub(r"@NUM", " ", x)
    x = re.sub(r"@PERCENT", " ", x)
    x = re.sub(r"@PERSON", " ", x)
    x = re.sub(r"@MONTH", " ", x)
    x = re.sub(r"@CITY", " ", x)
    x = re.sub(r"@YEAR", " ", x)
    x = re.sub(r"dear", " ", x)
    x = re.sub(r"local newspaper", " ", x)
    x = re.sub(r"dear newspaper", " ", x)
    x = re.sub(r'[^\w\s]', '', x)
    x = unidecode.unidecode(x)
    x = re.sub(r'\s+', ' ', x)
    x = re.sub(r'\d+', '', x)
    x = x.lower()
    x = re.sub(r"that's", "that is", x)
    x = re.sub(r"there's", "there is", x)
    x = re.sub(r"what's", "what is", x)
    x = re.sub(r"where's", "where is", x)
    x = re.sub(r"it's", "it is", x)
    x = re.sub(r"who's", "who is", x)
    x = re.sub(r"i'm", "i am", x)
    x = re.sub(r"she's", "she is", x)
    x = re.sub(r"he's", "he is", x)
    x = re.sub(r"they're", "they are", x)
    x = re.sub(r"who're", "who are", x)
    x = re.sub(r"you're", "you are", x)
    x = re.sub(r"ain't", "am not", x)
    x = re.sub(r"aren't", "are not", x)
    x = re.sub(r"wouldn't", "would not", x)
    x = re.sub(r"shouldn't", "should not", x)
    x = re.sub(r"couldn't", "could not", x)
    x = re.sub(r"doesn't", "does not", x)
    x = re.sub(r"isn't", "is not", x)
    x = re.sub(r"can't", "can not", x)
    x = re.sub(r"couldn't", "could not", x)
    x = re.sub(r"won't", "will not", x)
    x = re.sub(r"i've", "i have", x)
    x = re.sub(r"you've", "you have", x)
    x = re.sub(r"they've", "they have", x)
    x = re.sub(r"we've", "we have", x)
    x = re.sub(r"don't", "do not", x)
    x = re.sub(r"didn't", "did not", x)
    x = re.sub(r"i'll", "i will", x)
    x = re.sub(r"you'll", "you will", x)
    x = re.sub(r"he'll", "he will", x)
    x = re.sub(r"she'll", "she will", x)
    x = re.sub(r"they'll", "they will", x)
    x = re.sub(r"we'll", "we will", x)
    x = re.sub(r"i'd", "i would", x)
    x = re.sub(r"you'd", "you would", x)
    x = re.sub(r"he'd", "he would", x)
    x = re.sub(r"she'd", "she would", x)
    x = re.sub(r"they'd", "they would", x)
    x = re.sub(r"we'd", "we would", x)
    x = re.sub(r"she's", "she has", x)
    x = re.sub(r"he's", "he has", x)
    x = [lemmer.lemmatize(w) for w in x.split()]
    return ' '.join(x)


def count_spell_error(essay):
    clean_essay = re.sub(r'\W', ' ', str(essay).lower())
    clean_essay = re.sub(r'[0-9]', '', clean_essay)

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

# Delete the essays that have less than 200 words ~ approx. 20 essays

# Apply preprocessing onto both dataframes
df['Essay_Prep'] = df['Essay'].apply(preprocess)
df1['Essay_Prep'] = df1['essay'].apply(preprocess)

# Check for Mispelled Words in Original DataFrame (df1)
# df1['Spelling_Mistakes_Count'] = df1['Essay_Prep'].apply(count_spell_error)

print(df.shape, df1.shape)

# Split X & Y
X = df['Essay_Prep'].values
y = df['Essay Score'].values

print(y.shape, type(y), X.shape, type(X))

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Tokenize
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
word_dict = tokenizer.index_word

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

word_index = tokenizer.word_index
print("Found {0} unique words: ".format(len(word_index)))

# Padding
X_train_pad = pad_sequences(X_train_seq, maxlen=500, padding='post')    # change the maxlen
X_test_pad = pad_sequences(X_test_seq, maxlen=500, padding='post')      # change the maxlen
X_train_pad[:5]

print(X_train_pad.shape)

# LSTM Network
MAX_NB_WORDS = 8572
EMBEDDING_DIM = 100

lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=MAX_NB_WORDS+1, output_dim=500, input_length=X_train_pad.shape[1]))
lstm_model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2))
lstm_model.add(Dense(1, activation='linear'))

lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
lstm_model.summary()

# Setting up the Keras LSTM Network
# the code below is taken from https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17
# Building the model: --> Stuck on this
# model.add(SpatialDropout1D(0.2))
# model.add(Dense(13, activation='linear'))

epochs = 10
batch_size = 64

# Having issues with this line of code --> i need to figure out how to run 1 essay at a time instead of the whole corpus
history = lstm_model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

import matplotlib.pyplot as plt
training_loss = history.history['loss']
test_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss) + 1)

plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot the accuracy
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot((history.history['mean_squared_error']), 'r', label='train')
ax.plot((history.history['val_mean_squared_error']), 'b', label='val')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Mean Squared Error', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)



# accr = model.evaluate(X_test, y_test)
# print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

