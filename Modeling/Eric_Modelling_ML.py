import os
import textstat
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from Modeling.final_preprocessing import final_preprocessing, count_spell_error, regex_cleaning

os.getcwd()

os.chdir('/Users/ericross/School/Queens_MMAI/MMAI/MMAI_891/Project')

# Directories
DATASET_DIR = './asap-aes/'

##
# IMPORTING DATA

# Import Data For Spell Checker
data1 = pd.read_csv(os.path.join(DATASET_DIR, "training_set_rel3.tsv"), sep='\t', encoding='ISO-8859-1')
data1 = data1[12253:].copy()
data1 = pd.DataFrame(data=data1, columns=["essay", "domain1_score"])
data1 = data1.reset_index(drop=True)

# Import Data from Chin's Spell Checked Document
data = pd.read_csv(os.path.join(DATASET_DIR, "Essays_SpellCheck_Set8.csv"))
data = data.drop(['Unnamed: 0'], axis=1)
data.info()
data.head()

## PREPROCESSING STEPS:

df = final_preprocessing(data)
print(f'the df shape after feature engineering is: {df.shape}')

## VECTORIZING:

# Vectorizing the text and converting to columns (taken from Steve's Session 5): TF-IDF

vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.05, max_features=500, ngram_range=[1, 3])
dtm = vectorizer.fit_transform(df['Essay_Prep'])

vectorizer.get_feature_names()

bag_of_word_df = pd.DataFrame(dtm.toarray(), columns=vectorizer.get_feature_names(), index=df.index)

df_bow = pd.concat([df, bag_of_word_df], axis=1)
print(df_bow.shape)

##
# FEATURE ENGINEERING (#Taken from Steve's Lecture 5)

df_bow['Length'] = df_bow['Essay_Prep'].apply(lambda x: len(x))
df_bow['Syllable_Count'] = df_bow['Essay_Prep'].apply(lambda x: textstat.syllable_count(x))
df_bow['Flesch_Reading_Ease'] = df_bow['Essay_Prep'].apply(lambda x: textstat.flesch_reading_ease(x))
df_bow = df_bow.drop(['Essay_Prep', 'Essay'], axis=1)


##
# SPLIT X & Y FROM DATAFRAME

X = df_bow.drop(['Essay Score'], axis=1)
X = X.values
y = df_bow['Essay Score'].values

print(y.shape, type(y), X.shape, type(X))

##
# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


##
# MODEL CREATION

regr = LinearRegression()
regr.fit(X_train, y_train)

##
# PREDICT

y_pred = regr.predict(X_test)

##
# PERFORMANCE METRICS RESULTS
# The coefficients

print('Coefficients: \n', regr.coef_)

# The mean squared error

print('Mean absolute error: %.2f'
      % mean_absolute_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction

print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))
