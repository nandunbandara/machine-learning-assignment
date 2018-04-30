import data_preprocessing as dp
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier as rf


DATA_PATH = './data/dataset.csv'

# get preprocessed data sets
x_train, x_test, y_train, y_test = dp.preprocess_training_data(DATA_PATH)

# using a random forest classifier
classifier = rf(n_estimators=100, n_jobs=-1, max_features='auto')

classifier.fit(x_train, y_train)

print "\nPredicting..."
prediction = classifier.predict(x_test)

print "\n\nPrediction"
print ".........."
print "Number of correct predictions: %d/%d" % (accuracy_score(y_test, prediction, normalize=False), y_test.size)
print "Accuracy: %f" % (accuracy_score(y_test, prediction)*100)
print "Mean Squared Error: %f" % mean_squared_error(y_test, prediction)