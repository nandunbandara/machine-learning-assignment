# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.nan)

def preprocess_data(path):

    # Import dataset
    dataset = pd.read_csv(path)
    dataset.drop('Name', axis=1, inplace=True)
    dataset.drop('Ticket', axis=1, inplace=True)
    dataset.drop('Fare', axis=1, inplace=True)
    dataset.drop('Embarked', axis=1, inplace=True)

    # Features matrix
    X = dataset.loc[:, ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Cabin']]

    price_index = len(dataset.columns) - 1

    # Target matrix
    Y = dataset.loc[:, 'Survived']

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

    return X,Y


# onehotencoder = OneHotEncoder(categorical_features=[1])
# X[:, 1] = onehotencoder.fit_transform(X).toarray()





