import data_preprocessing as dp
import pandas as pd

TRAIN_DATA_PATH = './data/train.csv'
TEST_DATA_PATH = './data/test.csv'

x_train, x_test, y_train, y_test = dp.preProcessTrainingData(TRAIN_DATA_PATH)
# x_test = dp.preProcessTestData(TEST_DATA_PATH)

# using a random forest classifier (can be any classifier)
from sklearn.ensemble import RandomForestClassifier as rf

classifier = rf()
classifier.fit(x_train, y_train)

prediction = classifier.predict(x_test)

from sklearn.metrics import accuracy_score

print accuracy_score(y_test, prediction, normalize=False)
print accuracy_score(y_test, prediction)*100