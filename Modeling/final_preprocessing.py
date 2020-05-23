import re
import unidecode
import collections
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet

##
# Lemmatizer


def lem(x):
    lemmer = WordNetLemmatizer()
    x = [lemmer.lemmatize(w) for w in x.split()]
    return ' '.join(x)

##
# Regular Expression Cleaning


def regex_cleaning(x):
    # x = re.sub(r"[abc\\]$", "'", x)        # trying to get rid of \'s notation
    x = re.sub(r"@CAPS", " ", x)
    x = re.sub(r"@DATE", " ", x)
    x = re.sub(r"@LOCATION", " ", x)
    x = re.sub(r"@ORGANIZATION", " ", x)
    x = re.sub(r"@NUM", " ", x)
    x = re.sub(r"@PERCENT", " ", x)
    x = re.sub(r"@PERSON", " ", x)
    x = re.sub(r"pERSON", " ", x)               # Chin's spellchecker version
    x = re.sub(r" mon ", " mom ", x)            # need to fix the mispelled word of mom to mon in spell checked version
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
    x = re.sub(r"lifers", "life", x)            # Chin's spellchecker version
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
    x = re.sub(r"caps", " ", x)               # Chin's spellchecker version
    x = re.sub(r"location", " ", x)           # Chin's spellchecker version
    x = re.sub(r"date", " ", x)               # Chin's spellchecker version
    x = re.sub(r"person", " ", x)             # Chin's spellchecker version
    x = re.sub(r"organization", " ", x)       # Chin's spellchecker version
    return x


##
# Delete Essays With < 200 words = ~ approx. 20 essays --> need to figure this out


# def less_than_200(essay):
    # df['len_words'] = df['Essay_Prep'].apply(nltk.word_tokenize(['Essay_Prep']))
    # less_than_200_index = df[ df['len_words']].index
    # df.drop(less_than_200_index, inplace=True

##
# EVERYTHING BELOW IS TAKEN FROM: https://github.com/shubhpawar/Automated-Essay-Scoring

# Count Number of Incorrectly Spelled Words


def count_spell_error(essay):
    #   big.txt: It is a concatenation of public domain book excerpts from Project Gutenberg
    #   and lists of most frequent words from Wiktionary and the British National Corpus.
    #   It contains about a million words.
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
        if word not in word_dict:
            mispell_count += 1

    return mispell_count

##
# Calculating Average Word Length in an Essay

def avg_word_len(essay):

    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(essay)

    return sum(len(word) for word in words) / len(words)

##
# Calculating Number of Words in an Essay

def word_count(essay):

    clean_essay = re.sub(r'\W', ' ', essay)
    words = nltk.word_tokenize(clean_essay)

    return len(words)

##
# Calculating Number of Characters in an Essay

def char_count(essay):
    clean_essay = re.sub(r'\s', '', str(essay).lower())

    return len(clean_essay)

##
# Calculating Number of Sentences in an Essay

def sent_count(essay):

    sentences = nltk.sent_tokenize(essay)

    return len(sentences)


##
# Calculating Number of Lemmas Per Essay

def count_lemmas(essay):
    tokenized_sentences = nltk.word_tokenize(essay)

    lemmas = []
    wordnet_lemmatizer = WordNetLemmatizer()

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:

            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('J'):
                pos = wordnet.ADJ
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('V'):
                pos = wordnet.VERB
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            elif pos_tag.startswith('R'):
                pos = wordnet.ADV
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))
            else:
                pos = wordnet.NOUN
                lemmas.append(wordnet_lemmatizer.lemmatize(token_tuple[0], pos))

    lemma_count = len(set(lemmas))

    return lemma_count


##
# calculating number of nouns, adjectives, verbs and adverbs in an essay

def count_pos(essay):
    tokenized_sentences = nltk.word_tokenize(essay)

    noun_count = 0
    adj_count = 0
    verb_count = 0
    adv_count = 0

    for sentence in tokenized_sentences:
        tagged_tokens = nltk.pos_tag(sentence)

        for token_tuple in tagged_tokens:
            pos_tag = token_tuple[1]

            if pos_tag.startswith('N'):
                noun_count += 1
            elif pos_tag.startswith('J'):
                adj_count += 1
            elif pos_tag.startswith('V'):
                verb_count += 1
            elif pos_tag.startswith('R'):
                adv_count += 1

    return noun_count, adj_count, verb_count, adv_count

##
# Compile All Of The Pre-processing & Feature Engineering


def final_preprocessing(data):
    df = data.copy()
    #df1 = data1.copy()

    # 1) Apply Regular Express Cleaning on both dataframes
    df['Essay_Prep'] = df['Essay'].apply(regex_cleaning)
    df1['Essay_Prep'] = df1['essay'].apply(regex_cleaning)

    # 2) Append Mispelled Words in df1
    df1['Spelling_Mistakes_Count'] = df1['Essay_Prep'].apply(count_spell_error)
    df1 = pd.DataFrame(data1['Spelling_Mistakes_Count'])
    df = pd.concat([df, df1], axis=1)

    # 4) Append Character Counter
    df['Char_Count'] = df['Essay_Prep'].apply(char_count)

    # 5) Append Word Counter
    df['Word_Count'] = df['Essay_Prep'].apply(word_count)

    # 6) Append Sentence Counter
    df['Sent_Count'] = df['Essay_Prep'].apply(sent_count)

    # 7) Append Average Word Counter
    df['Avg_Word_Count'] = df['Essay_Prep'].apply(avg_word_len)

    # 8) Append Lemma Counter  --> Takes too long to run
    df['Lemma_Count'] = df['Essay_Prep'].apply(count_lemmas)

    # 9) Append Noun/Adjective/Verb/Adverb Counter --> Takes too long to run
    df['noun_count'], df['adj_count'], df['verb_count'], \
    df['adv_count'] = zip(*df['Essay_Prep'].map(count_pos))

    # 3) Apply Lematizer
    df['Essay_Prep'] = df['Essay_Prep'].apply(lem)

    return df


