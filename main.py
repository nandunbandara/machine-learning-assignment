import data_preprocessing as dp

TRAIN_DATA_PATH = './data/train.csv'
TEST_DATA_PATH = './data/test.csv'

x_train, y_train = dp.preprocess_data(TRAIN_DATA_PATH)

print x_train