import pandas as pd
import pickle
import helper
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import confusion_matrix


# Set file names
test_file = "preprocess_test.csv"
model_file = "LSVC.pickle"

# Read test data
print("Reading test data")
test = pd.read_csv(test_file,encoding='utf-8-sig',delimiter=',')

# Get target (what we want to predict)
# and features (list of features to help predict)

target, features = helper.format_dataframe(test)

# Load classifier pickle file
print("Loading the classifier")
classifier = pickle.load(open(model_file))

# Make prediction
print("Making predictions")

transformer = TfidfTransformer()
features = transformer.fit_transform(features)
predictions = classifier.predict(features)

# Evaluation Metrics
accuracy = metrics.accuracy_score(target, predictions)
precision = metrics.precision_score(target, predictions, average='macro')
recall = metrics.recall_score(target, predictions, average='macro')
f1 = metrics.f1_score(target, predictions, average='macro')
print "Accuracy: %f" % accuracy
print "Precision: %f" % precision
print "Recall: %f" % recall
print "F1: %f" % f1
print confusion_matrix(target, predictions)