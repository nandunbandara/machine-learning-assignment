# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

np.set_printoptions(threshold=np.nan)

def preProcessTrainingData(path):
    # Import dataset
    dataFrame = pd.read_csv(path)
    include = ['URL', 'URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS','CHARSET',
               'SERVER','CONTENT_LENGTH','WHOIS_COUNTRY','WHOIS_STATEPRO',
               'WHOIS_REGDATE','WHOIS_UPDATED_DATE','TCP_CONVERSATION_EXCHANGE',
               'DIST_REMOTE_TCP_PORT','REMOTE_IPS','APP_BYTES','SOURCE_APP_PACKETS',
               'REMOTE_APP_PACKETS','APP_PACKETS','DNS_QUERY_TIMES','Type']
    dataFrame_ = dataFrame[include]

    categoricals = []
    for col, col_type in dataFrame_.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        # else:
        #     dataFrame_[col].fillna(0, inplace=True)

    dataFrameEncoded = pd.get_dummies(dataFrame_, columns=categoricals, dummy_na=True)

    # set the mean value of the column for nan
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(dataFrameEncoded)
    dataFrameEncoded[:] = imputer.transform(dataFrameEncoded[:])

    dependent_variable = 'Type'

    x = dataFrameEncoded[dataFrameEncoded.columns.difference([dependent_variable])]
    y = dataFrameEncoded[dependent_variable]

    # Scale the age variable
    # scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    # age = x[['Age']].values.astype(int)
    # age_scaled = scaler.fit_transform(age)
    # x['Age'] = age_scaled


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return x_train, x_test, y_train, y_test

