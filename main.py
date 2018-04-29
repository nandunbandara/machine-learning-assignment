import data_preprocessing as dp
from sklearn.metrics import accuracy_score, mean_squared_error


DATA_PATH = './data/dataset.csv'

x_train, x_test, y_train, y_test = dp.preProcessTrainingData(DATA_PATH)

# using a random forest classifier
from sklearn.ensemble import RandomForestClassifier as rf

classifier = rf(n_estimators=100, criterion='gini', min_samples_leaf=1, bootstrap=True, oob_score=True,
                n_jobs=1, random_state=2, warm_start=False, class_weight=None)

classifier.fit(x_train, y_train)

print "\nPredicting..."
prediction = classifier.predict(x_test)

print "\n\nPrediction"
print ".........."
print "Number of correct predictions: %d/%d" % (accuracy_score(y_test, prediction, normalize=False), y_test.size)
print "Accuracy: %f" % (accuracy_score(y_test, prediction)*100)
print "Mean Squared Error: %f" % mean_squared_error(y_test, prediction)