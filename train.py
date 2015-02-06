from __future__ import division

import pandas as pd
import pickle
import helper
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

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

    # Train Classifier
    print("Training the Classifier")

    """
    *~* Pick your classifier here *~*
    """
    classifier = LinearSVC() # SVM
    #classifier = GaussianNB() # Naive Bayes
    #classifier = KNeighborsClassifier(n_neighbors=3) # KNN

    # Cross-Validation
    mean_accuracy = 0.0
    mean_recall = 0.0
    mean_f1 = 0.0
    n = 10 # no. of fold for validation
    SEED = 42  # always use a seed for randomized procedures
    for i in range(n):
    	# for each iteration, randomly hold out 20% of the data as CV set
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
            features, target, test_size=.20, random_state=i*SEED)
        classifier.fit(X_train, y_train)
        preds = classifier.predict(X_cv)

        # Evaluation metrics
        recall = metrics.recall_score(y_cv, preds)
        accuracy = metrics.accuracy_score(y_cv, preds)
        f1 = metrics.f1_score(y_cv, preds)
        print "Accuracy (fold %d/%d): %f" % (i + 1, n, accuracy)
        print "Recall (fold %d/%d): %f" % (i + 1, n, recall)
        print "F1 (fold %d/%d): %f" % (i + 1, n, f1)
        print "*~*~*~*~*~*~*~*~"
        mean_accuracy += accuracy
        mean_recall += recall
        mean_f1 += f1	

    print "Mean Accuracy: %f" % (mean_accuracy/n)
    print "Mean Recall: %f" % (mean_recall/n)
    print "Mean F1: %f" % (mean_f1/n)


    # Save CLassifier as pickle file
    print("Saving the classifier")
    pickle.dump(classifier, open(model_file, "w"))

    
    # Result for dev file
    dev = pd.read_csv(dev_file)
    target_dev, features_dev = helper.format_dataframe(dev)
    predictions_dev = classifier.predict(features_dev)

    # Evaluation Metrics
    print "==============="
    print " DEV FILE"
    accuracy = metrics.accuracy_score(target_dev, predictions_dev)
    recall = metrics.recall_score(target_dev, predictions_dev)
    f1 = metrics.f1_score(target_dev, predictions_dev)
    print "Accuracy: %f" % accuracy
    print "Recall: %f" % recall
    print "F1: %f" % f1
    


main()