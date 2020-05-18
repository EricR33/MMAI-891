import nltk
import re
import pandas as pd
import os

pwd

# Set the current working directory
cd '/Users/ericross/School/Queens_MMAI/MMAI/MMAI_891/Project'

# Directories
DATASET_DIR = './asap-aes/'
WORKBOOK_DIR = './Courses/'

# Import .tsv file into Data Frame
df = pd.read_csv(os.path.join(DATASET_DIR, "training_set_rel3.tsv"), sep='\t', encoding='ISO-8859-1')

# Ensure that that the initial shape represents the actual file
df.head()
df.shape

# Downsize the dataframe to only include columns up to and including column 6 "domain1_score"
df = df.iloc[:, 0:7]
df.head()
df.shape

# Assign the y variable to 'domain1_score', check the shape and head of the y variable
y = df['domain1_score']
y.head()
y.shape

# Assign the X variable to columns = 'essay_id', 'essay_set', and 'essay'. Check the shape, head, description
X = df.iloc[:, 0:3]
X.head()
X.shape
X.describe

# Pandas Profiler Report
# pandas_profiling.ProfileReport(df) --> can't get to work in PyCharm


# for index, row in df.iterrows():
#    print(index, row)

# Regular Expression Cleaning
X_essay = X['essay']
X_test = X_essay.iloc[12259]
X_test = nltk.sent_tokenize(X_test)

for i in range(len(X_test)):
    X_test[i] = re.sub(r'\d', '', X_test[i])    # cleans out digits
    X_test[i] = re.sub(r'\s+', ' ', X_test[i])  # cleans out multiple spaces


sentence = """Welcome to the year 2018. Just ~%* ++++--- arrived at @Jack's place. #fun.
            I                  love                u"""
sentence = nltk.sent_tokenize(sentence)












paragraph = """Thank you all so very much. Thank you to the Academy. 
               Thank you to all of you in this room. I have to congratulate 
               the other incredible nominees this year."""

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


# Convert the model to an array
X = np.asarray(tfidf_matrix)
X = np.transpose(X)

# N-gram Modelling (Character Based)

text = """Global warming or climate change has become a worldwide concern. It is gradually developing into an 
        unprecedented environmental crisis evident in melting glaciers, changing weather patterns, rising sea levels, 
        floods, cyclones and droughts. Global warming implies an increase in the average temperature of the Earth due 
        to entrapment of greenhouse gases in the earth’s atmosphere."""

n = 3   # number of ngrams = trigram is 3

ngrams = {}
# Create the n-grams
for i in range(len(text) - n):          # Iterate through the characters in the list
    gram = text[i:i+n]                  # set the gram equal to 3 characters in sequence within the text
    print(f'ngram is: {gram}; ngrams dictionary is: {ngrams}')
    if gram not in ngrams.keys():       # creating an empty list if the gram is not within the ngrams dictionary keys
        ngrams[gram] = []
    ngrams[gram].append(text[i+n])      # appending the character that appears after the ngram to the as the value that relates to the ngram dictionary key(gram)

# N-Gram Testing
current_gram = text[0:n]        # This is selecting a n-gram
result = current_gram           # Sets the result equal to current gram
for i in range(100):
    if current_gram not in ngrams.keys():   # We want to break out of the loop if the current gram is not in the dictionary
        break
    possibilities = ngrams[current_gram]    # Returns the dictionary values for the current gram key within the ngrams dictionary
    print(possibilities)
    next_item = possibilities[random.randrange(len(possibilities))]         # randomly selects a character within the dictionary values relating to the current gram key
    result += next_item         # appends the randomly selected character to the current gram
    current_gram = result[len(result) - n:len(result)]      # This returns the current gram after the new character was added to the result
    # Example --> current_gram is 'Glo'. next_item is selected as 'b'. 'b' is appended to result to form 'Glob'
    # current_gram is then set to 'lob' and the process continues

print(result)


# N-Gram Modelling - (Word Based)
text = """Global warming or climate change has become a worldwide concern. It is gradually developing into an 
        unprecedented environmental crisis evident in melting glaciers, changing weather patterns, rising sea levels, 
        floods, cyclones and droughts. Global warming implies an increase in the average temperature of the Earth due 
        to entrapment of greenhouse gases in the earth’s atmosphere."""

