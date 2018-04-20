import data_preprocessing as dp
import pandas as pd

TRAIN_DATA_PATH = './data/train.csv'
TEST_DATA_PATH = './data/test.csv'

x_train, y_train = dp.preProcessTrainingData(TRAIN_DATA_PATH)
x_test = dp.preProcessTestData(TEST_DATA_PATH)

# using a random forest classifier (can be any classifier)
from sklearn.ensemble import RandomForestClassifier as rf

clf = rf()
clf.fit(x_train, y_train)

prediction = clf.predict(x_test)

print prediction