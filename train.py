from __future__ import division

import pandas as pd
import numpy as np
import pickle
import helper
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV

def main():
    """
    *~* SET FILE NAMES HERE *~*
    """
    train_file = "preprocess_train.csv"
    dev_file = "preprocess_dev.csv"
    model_file = "LSVC.pickle"

    # Read training csv data
    train = pd.read_csv(train_file)

    # Get target (what we want to predict)
    # and features (list of features to help predict)
    target, features = helper.format_dataframe(train)

    transformer = TfidfTransformer()
    features = transformer.fit_transform(features)

    # Train Classifier
    print("Training the Classifier")

    """
    *~* Pick your classifier here *~*
    """
    classifier = LinearSVC(C=100) # SVM
    #classifier = GaussianNB() # Naive Bayes
    #classifier = KNeighborsClassifier(n_neighbors=3) # KNN

    # Cross-Validation
    mean_accuracy = 0.0
    mean_recall = 0.0
    mean_f1 = 0.0
    n = 10 # no. of fold for validation
    SEED = 42  # always use a seed for randomized procedures
    """
    # Feature Selection
    # Option 1. Removing features with low variance
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    features = sel.fit_transform(features)

    # Option 2. Univariate feature selection
    features = SelectPercentile(chi2, percentile=10).fit_transform(features, target)

    # Option 3. Tree-based feature selection
    clf = ExtraTreesClassifier()
    features = clf.fit(features, target).transform(features)

    # Tune Classifier
    # Grid Search
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    classifier = GridSearchCV( SVC(), tuned_parameters, score_func='f1')
    classifier.fit(features, target)
    print classifier.best_estimator_
    """


    # TRAIN THE CLASSIFIER!!!
    classifier.fit(features, target)

    # Output prediction result for training data
    predictions_train = classifier.predict(features)
    np.savetxt('predict_train.txt',predictions_train,fmt="%s")


    # Save CLassifier as pickle file
    print("Saving the classifier")
    pickle.dump(classifier, open(model_file, "w"))

    # Evaluation Metrics for TRAIN
    print "==============="
    print " TRAIN FILE"
    accuracy = metrics.accuracy_score(target, predictions_train)
    precision = metrics.precision_score(target, predictions_train, average='macro')
    recall = metrics.recall_score(target, predictions_train, average='macro')
    f1 = metrics.f1_score(target, predictions_train, average='macro')
    print "Accuracy: %f" % accuracy
    print "Precision: %f" % precision
    print "Recall: %f" % recall
    print "F1: %f" % f1
    print confusion_matrix(target, predictions_train)
    
    # Result for dev file
    dev = pd.read_csv(dev_file)
    target_dev, features_dev = helper.format_dataframe(dev)
    features_dev = transformer.fit_transform(features_dev)
    predictions_dev = classifier.predict(features_dev)
    np.savetxt('predict_dev.txt',predictions_dev,fmt="%s")

    # Evaluation Metrics for DEV
    print "==============="
    print " DEV FILE"
    accuracy = metrics.accuracy_score(target_dev, predictions_dev)
    precision = metrics.precision_score(target_dev, predictions_dev, average='macro')
    recall = metrics.recall_score(target_dev, predictions_dev, average='macro')
    f1 = metrics.f1_score(target_dev, predictions_dev, average='macro')
    print "Accuracy: %f" % accuracy
    print "Precision: %f" % precision
    print "Recall: %f" % recall
    print "F1: %f" % f1
    print confusion_matrix(target_dev, predictions_dev)
    


main()