# MMAI-891
Natural Language Processing:
This GitHub repository is dedicated to the development of an NLP solution for grading essay submissions.

The final submission code files for MMAI-891 NLP are the following:

1) Modelling / GLOVE_LSTM_NN_Model.py   --> This is the final recommended NN for the project
2) Modelling / Final_Preprocessing.py   --> This is the combined preprocessing file
3) Modelling / LSTM_NN_Model.py         --> This is the baseline NN model
4) Modelling / NN_Models.py             --> This contains all NN models including the final solution except for #3 file
5) Modelling / ML_Models.py             --> This is the Linear Regression & Gradient Boosting Regressor models
6) Modelling / training_2-LSTM.zip      --> This is the saved weights for the GLOVE_LSTM_NN_Model.py file


The final submission data files for MMAI-891 NLP are the following:

1) Modelling / big.txt                    --> This is the dictionary used to calculate the number of mispelled words
2) Modelling / Essays_SpellCheck_Set8.csv --> This is the output csv file from the spellchecker
3) Modelliong / new_df-5.csv              --> This is the standardized csv data file after preprocessing
4) Modelling / training_set_rel3.csv      --> This is the raw csv data file before any preprocessing (Kaggle dataset)
5) Modelling / test_set_2_preprocessed.csv  --> This is the preprocessed Essay Set #2 used to test model generalization



Instructions to run GLOVE_LSTM_NN_Model.py File:

1) Recommend to do all of this in Google Colab due to processing time
2) Save the clean version of Kaggle dataset (Modelling / new_df-5.csv) in "gdrive/My Drive/MMAI891/" in Google Drive; unzip Modelling /  training_2-LSTM.zip and save the weights in "gdrive/My Drive/MMAI891/training_2/" in Google Drive
3) Run this code inside Google Colab. There might be slight differences if attempted to run on a Python IDE
4) All attempted models are saved in  Modelling / NN_Models.py


Instructions to run Final_Preprocessing.py

1) Change the directory on line 330 to the GitHub cloned repo "Modelling" folder
2) Run code --> This will take roughly 5 minutes in PyCharm
3) The output of this python file is the "new_df-5.csv" file, which is used in all of the modelling


Instructions to run ML_Models.py

1) Change the directory on line 12 to the GitHub cloned repo "Modelling" folder
2) Run code
3) Note --> Gradient Boosting Regressor has many RunTimeWarnings, but will run fully if given enough time (roughly < 10 minutes)


Instructions to run LSTM_NN_Model.py

1) Change the directory on line 14 to the GitHub cloned repo "Modelling" folder
2) Downgrade tensorflow to version 1.14
3) Run code
