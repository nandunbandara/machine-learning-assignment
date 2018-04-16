# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.nan)

# Import dataset
dataset = pd.read_csv('./data/train.csv')
dataset.drop('Id', axis=1, inplace=True)

# Features matrix
X = dataset.iloc[:, :-1].values

price_index = len(dataset.columns) - 1

# Target matrix
Y = dataset.iloc[:, price_index].values

# Check if a column is categorical or not
def is_categorical(column):
    return column.dtype.name == 'category' or column.dtype.name == 'object'

def is_numerical(column):
    return column.dtype.name == 'int64'

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

for column in dataset:
    if is_categorical(dataset[column]):
        index = dataset.columns.get_loc(column)
        X[:, index] = labelencoder_X.fit_transform(X[:, index])


# Handle missing/null data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X)
X = imputer.transform(X)


# Feature scaling
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X = ss_X.fit_transform(X)
X = ss_X.transform(X)


# onehotencoder = OneHotEncoder(categorical_features=[1])
# X[:, 1] = onehotencoder.fit_transform(X).toarray()

print(X)



