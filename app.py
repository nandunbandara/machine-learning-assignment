import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import data_preprocessing as dp

TRAIN_DATA_PATH = './data/train.csv'
TEST_DATA_PATH = './data/test.csv'

x_train, y_train = dp.preprocess_data(TRAIN_DATA_PATH)
x_test, y_test = dp.preprocess_data(TEST_DATA_PATH)


regressor = LinearRegression()
model = regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print ('Score: %.2f' % model.score(x_test, y_test))
print ('Variance Score: %.2f' % r2_score(y_test, y_pred))