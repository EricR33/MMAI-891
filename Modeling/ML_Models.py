import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

os.getcwd()

os.chdir('/Users/ericross/School/Queens_MMAI/MMAI/PyCharm Projects/MMAI-891/Modeling')


##
# IMPORTING SET #8 PREPROCESSED FILE
data_8 = pd.read_csv("new_df-5.csv")

data_8 = data_8.drop(['Unnamed: 0'], axis=1)
data_8.info()
data_8.head()

##
# IMPORT SET #2 PREPROCESSED FILE
data_2 = pd.read_csv("test_set_2_preprocessed.csv")

data_2 = data_2.drop(['Unnamed: 0'], axis=1)
data_2.info()
data_2.head()
list(data_2)


## VECTORIZING SET#8:

# Vectorizing the text and converting to columns (taken from Steve's Session 5): TF-IDF

vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.05, max_features=500, ngram_range=[1, 3])
dtm_8 = vectorizer.fit_transform(data_8['Essay_Prep'])

feature_names_cv_8 = vectorizer.get_feature_names()

bag_of_word_df8 = pd.DataFrame(dtm_8.toarray(), columns=feature_names_cv_8, index=data_8.index)

df8_bow = pd.concat([data_8, bag_of_word_df8], axis=1)
print(df8_bow.shape)


## VECTORIZING SET#2:

# Vectorizing the text and converting to columns (taken from Steve's Session 5): TF-IDF

vectorizer = TfidfVectorizer(max_df=0.5, min_df=0.05, max_features=500, ngram_range=[1, 3])
dtm_2 = vectorizer.fit_transform(data_2['Essay_Prep'])

feature_names_cv_2 = vectorizer.get_feature_names()

bag_of_word_df2 = pd.DataFrame(dtm_2.toarray(), columns=feature_names_cv_2, index=data_2.index)

df2_bow = pd.concat([data_2, bag_of_word_df2], axis=1)
print(df2_bow.shape)

##
# Drop Unstructured Features & Normalize Grading Scale (essay set 8 is out of 60 and set 2 was out of 6)

df8_bow = df8_bow.drop(['Essay_Prep', 'Essay'], axis=1)
df2_bow = df2_bow.drop(['Essay_Prep', 'Essay'], axis=1)
df2_bow['Essay Score'] = df2_bow['Essay Score'] * 10
##
# SPLIT X & Y FROM DATAFRAME

X = df8_bow.drop(['Essay Score'], axis=1)
X = X.values
y = df8_bow['Essay Score'].values

print(y.shape, type(y), X.shape, type(X))

##
# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


##
# LINEAR REGRESSION MODEL CREATION

regr = LinearRegression()
regr.fit(X_train, y_train)

##
# PREDICT

y_pred = regr.predict(X_test)

##
# PERFORMANCE METRICS RESULTS

# The mean absolute error

print('Mean absolute error: %.2f'
      % mean_absolute_error(y_test, y_pred))


##
# GRADIENT BOOSTING REGRESSION --> TAKEN FROM https://github.com/shubhpawar/Automated-Essay-Scoring/

params = {'n_estimators':[100, 1000], 'max_depth':[2], 'min_samples_split': [2],
          'learning_rate':[3, 1, 0.1, 0.3], 'loss': ['ls']}

gbr = ensemble.GradientBoostingRegressor()

grid = GridSearchCV(gbr, params)
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_)

mae = mean_absolute_error(y_test, y_pred)
print("MAE: %.4f" % mae)


##
# Test the Generalization Ability
X2 = pd.DataFrame(df2_bow.iloc[:723])
X2.shape
X2 = df2_bow.drop(['Essay Score'], axis=1)

X2 = X2.values

y2 = df2_bow['Essay Score'].values

y2_pred = grid.predict(X2)

mae = mean_absolute_error(y2, y2_pred)
print("MAE: %.4f" % mae)