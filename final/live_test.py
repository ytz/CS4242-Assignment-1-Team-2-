import twitter_data
import data_cleaning
import preprocess
import helper
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

# Name of file to be tested
file_to_test = "live.csv" #EDIT THIS FIELD

""" Data Cleaning """
print 'Data Cleaning'
api = twitter_data.getAPI()
data_cleaning.cleanData(file_to_test, 'fixed_'+file_to_test,api)

""" Preprocess """
print 'Preprocess'
df_train = pd.read_csv('preprocess_train.csv')
df_test = pd.read_csv('fixed_'+file_to_test)
df_test = preprocess.copy_features(df_train, df_test)

""" Predict """
print 'Predict'
model_file = "LSVC.pickle"

# Load classifier pickle file
print("Loading the classifier")
classifier = pickle.load(open(model_file))

# Make prediction
print("Making predictions")
target, features = helper.format_dataframe(df_test)
transformer = TfidfTransformer()
features = transformer.fit_transform(features)
predictions = classifier.predict(features)

# Save predictions
np.savetxt('predict_live.txt',predictions,fmt="%s")