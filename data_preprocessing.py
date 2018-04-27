# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.nan)

def preProcessTrainingData(path):
    # Import dataset
    dataFrame = pd.read_csv(path)
    include = ['Age', 'Sex', 'Embarked', 'Survived']
    dataFrame_ = dataFrame[include]

    categoricals = []
    for col, col_type in dataFrame_.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            dataFrame_[col].fillna(0, inplace=True)

    dataFrameEncoded = pd.get_dummies(dataFrame_, columns=categoricals, dummy_na=True)

    dependent_variable = 'Survived'
    x = dataFrameEncoded[dataFrameEncoded.columns.difference([dependent_variable])]
    y = dataFrameEncoded[dependent_variable]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return x_train, x_test, y_train, y_test

def preProcessTestData(path):
    # Import dataset
    dataFrame = pd.read_csv(path)

    dataFrame_ = dataFrame[['Age', 'Sex', 'Embarked']]

    categoricals = []
    for col, col_type in dataFrame_.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            dataFrame_[col].fillna(0, inplace=True)

    dataFrameEncoded = pd.get_dummies(dataFrame_, columns=categoricals, dummy_na=True)


    return dataFrameEncoded






