import pandas as pd
import pickle
import helper
from sklearn import metrics

# Set file names
test_file = "preprocess_test.csv"
model_file = "LSVC.pickle"

# Read test data
print("Reading test data")
test = pd.read_csv(test_file)

# Get target (what we want to predict)
# and features (list of features to help predict)

target, features = helper.format_dataframe(test)

# Load classifier pickle file
print("Loading the classifier")
classifier = pickle.load(open(model_file))

# Make prediction
print("Making predictions")
predictions = classifier.predict(features)

# Evaluation Metrics
accuracy = metrics.accuracy_score(target, predictions)
recall = metrics.recall_score(target, predictions, average='macro')
f1 = metrics.f1_score(target, predictions, average='macro')
print "Accuracy: %f" % accuracy
print "Recall: %f" % recall
print "F1: %f" % f1