print("Hello world!")

# Might need these
#from nltk.stem import WordNetLemmatizer
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
# TODO: import other libraries as necessary

###############################
# LOAD DATA
###############################
import pandas as pd
import os

DATASET_DIR = "./Data/"
X = pd.read_csv(os.path.join(DATASET_DIR, 'training_set_rel3.tsv'), sep = '\t', encoding='ISO-8859-1')

Y = X['domain1_score']

X.head()

####################################################
# PREPROCESSING THE DATA
#
# Convert essays to feature vectors and take out:
# 1 - Stopwords
# 2 -
####################################################
import numpy as np
# import re for regular expression library
import nltk
import re
from nltk.corpus import stopwords

def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    index2word_set = set(model.wv.index2word)
    for word in words:
        if word in index2word_set:
            num_words += 1
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs


####################################################
# Run Data preprocessing functions
####################################################

train_essays = X['essay']
sentences = []

for essay in train_essays:
            # Obtaining all sentences from the training essays.
            sentences += essay_to_sentences(essay, remove_stopwords = True)

print(sentences[2])

####################################################
# CREATE A WORD CLOUD
# disclaimer - warning: shelley code! :)
# requires seaborn, unidecode, wordcloud, lxml, html5lib
####################################################

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# import re for regular expression library
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import unidecode
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

####################################################
# LOAD YOUR FILE/DATA INTO A PANDAS DATAFRAME
####################################################

sl_df = pd.read_csv("/Users/shelley/Documents/ShelleyLI_CSV.csv")
#print(sl_df.info())
#print(sl_df.head())


#### testing code only IGNORE ####
from bs4 import BeautifulSoup

html = "This is a test"
soup = BeautifulSoup(html,"html.parser")
text = soup.get_text(strip=True)
print (text)

####################################################
# CLEAN YOUR DATA
####################################################

from nltk.corpus import stopwords
from bs4 import BeautifulSoup

stop_words = set(stopwords.words('english'))
stop_words.add('for')
stop_words.add('see')
###################################################
# REMOVED EXTRA WORDS
####################################################

stop_words.add('endorsement')
stop_words.add('endorsements')
stop_words.add('addition')
stop_words.add('please')
stop_words.add('resume')
stop_words.add('individual')
stop_words.add('filled')
stop_words.add('told')
stop_words.add('spent')
stop_words.add('find')
stop_words.add('divisional')
stop_words.add('would')
stop_words.add('didnt')
stop_words.add('etc')
stop_words.add('undergraduate')
stop_words.add('last')
stop_words.add('entire')
stop_words.add('go')
stop_words.add('school')
stop_words.add('department')
stop_words.add('bachelor')
stop_words.add('divisional')
stop_words.add('role')
stop_words.add('smith')
stop_words.add('even')
stop_words.add('together')
stop_words.add('evening')
stop_words.add('enclosed')
stop_words.add('full')
stop_words.add('part')
stop_words.add('direct')
stop_words.add('basis')
stop_words.add('including')
stop_words.add('perfect')
stop_words.add('large')
stop_words.add('time')
stop_words.add('desktop')
stop_words.add('drove')
stop_words.add('company')
stop_words.add('take')
stop_words.add('high')
stop_words.add('coupled')
stop_words.add('many')
stop_words.add('interim')
stop_words.add('background')
stop_words.add('year')
stop_words.add('know')
stop_words.add('hands')
stop_words.add('candidate')
stop_words.add('career')
stop_words.add('add')
stop_words.add('cornerstone')
stop_words.add('earned')
stop_words.add('degree')
stop_words.add('brand')
stop_words.add('aspect')
stop_words.add('following')
stop_words.add('second')
stop_words.add('application')
stop_words.add('received')
stop_words.add('next')
stop_words.add('someone')
stop_words.add('honor')
stop_words.add('dtype')
stop_words.add('sent_clean')
stop_words.add('object')
stop_words.add('Length')
stop_words.add('Name')

lemmer = WordNetLemmatizer()
porter = PorterStemmer()
lancaster = LancasterStemmer()


def preprocess(x):
    # Remove HTML tags
    x = BeautifulSoup(x,"html.parser").get_text(strip=True)

    # Lower case
    x = x.lower()

    # Remove punctuation
    x = re.sub(r'[^\w\s]', '', x)

    # Remove non-unicode
    x = unidecode.unidecode(x)

    # Remove numbers
    x = re.sub(r'\d+', '', x)

    # Remove stopwords and lemmatize
    x = [lemmer.lemmatize(w) for w in x.split() if w not in stop_words]

    # Stemming
    # x = [porter.stem(w) for w in x.split() if w not in stop_words]

    return ' '.join(x)


####################################################
# CREATE A LIST THAT YOU WANT TO VISUALIZE
####################################################
list(sl_df)
#sl_df.shape
#sl_df.head()
#sl_df.tail()

#sl_df['sent_clean'] = sl_df['Sentence'].apply(preprocess)
#sl_df.shape
#sl_df.head()
#sl_df.tail()
#sl_df.count()

print(sl_df['Sentence'])
sl_df['sent_clean'] = sl_df['Sentence'].apply(preprocess)

####################################################
# CREATE THE WORD CLOUD
####################################################

# https://www.datacamp.com/community/tutorials/wordcloud-python

def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1  # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()



show_wordcloud(sl_df['sent_clean'])
