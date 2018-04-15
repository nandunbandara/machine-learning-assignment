# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('./data/train.csv')
dataset.drop('Id', axis=1, inplace=True)

# Features matrix
X = dataset.iloc[:, :-1].values

price_index = len(dataset.columns) - 1

# Target matrix
Y = dataset.iloc[:, price_index].values

# Handle missing/null data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imputer = imputer.fit(X)
# X = imputer.transform(X)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# onehotencoder = OneHotEncoder(categorical_features=[1])
# X[:, 1] = onehotencoder.fit_transform(X).toarray()


print(X)


# Check if a column is categorical or not
def is_categorical(column):
    return column.dtype.name == 'category'

