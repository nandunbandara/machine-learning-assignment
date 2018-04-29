from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import numpy as np



def compare_classifiers(x_train, x_test, y_train, y_test):

    names = ["Nearest Neighbors",
             "Decision Tree", "Neural Net",
             "Naive Bayes"]


    classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1),
        GaussianNB()
    ]
    for name, clf in zip(names, classifiers):
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)

        if hasattr(clf, "decision_function"):
            prediction = clf.decision_function(x_test)
        else:
            prediction = clf.predict(x_test)

        print "\n\nPrediction of "+name
        print ".........."
        print "Number of correct predictions: %d/%d" % (
        accuracy_score(y_test, prediction, normalize=False), y_test.size)
        print "Accuracy: %f" % (accuracy_score(y_test, prediction) * 100)
        print "Mean Squared Error: %f" % mean_squared_error(y_test, prediction)

    plt.tight_layout()
    plt.show()