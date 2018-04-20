# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(threshold=np.nan)

def preProcessTrainingData(path):
    # Import dataset
    df = pd.read_csv(path)
    # df.drop('Name', axis=1, inplace=True)
    # df.drop('Ticket', axis=1, inplace=True)
    # df.drop('Fare', axis=1, inplace=True)
    # df.drop('Embarked', axis=1, inplace=True)
    include = ['Age', 'Sex', 'Embarked', 'Survived']
    df_ = df[include]

    categoricals = []
    for col, col_type in df_.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)

    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

    dependent_variable = 'Survived'
    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    return x,y

def preProcessTestData(path):
    # Import dataset
    df = pd.read_csv(path)
    # df.drop('Name', axis=1, inplace=True)
    # df.drop('Ticket', axis=1, inplace=True)
    # df.drop('Fare', axis=1, inplace=True)
    # df.drop('Embarked', axis=1, inplace=True)

    df_ = df[['Age', 'Sex', 'Embarked']]

    categoricals = []
    for col, col_type in df_.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)

    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)


    return df_ohe






