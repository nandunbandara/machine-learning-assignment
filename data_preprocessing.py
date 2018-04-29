# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import datetime as dt

np.set_printoptions(threshold=np.nan)

def preProcessTrainingData(path):
    # Import dataset
    dataFrame = pd.read_csv(path)
    include = ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS',
               'CONTENT_LENGTH','WHOIS_COUNTRY','WHOIS_STATEPRO',
               'WHOIS_REGDATE','WHOIS_UPDATED_DATE','TCP_CONVERSATION_EXCHANGE',
               'DIST_REMOTE_TCP_PORT','REMOTE_IPS','APP_BYTES','SOURCE_APP_PACKETS',
               'REMOTE_APP_PACKETS','APP_PACKETS','DNS_QUERY_TIMES','Type']
    dataFrame_ = dataFrame[include]

    categoricals = []
    for col, col_type in dataFrame_.dtypes.iteritems():
        if col_type == 'O':
            categoricals.append(col)
        else:
            dataFrame_[col].fillna(0, inplace=True)

    # dataFrame_['WHOIS_REGDATE'] = np.where(dataFrame_['WHOIS_REGDATE'] == 'None','19/03/2017 0:00', dataFrame_['WHOIS_REGDATE'])
    # dataFrame_['WHOIS_UPDATED_DATE'] = np.where(dataFrame_['WHOIS_UPDATED_DATE'] == 'None', '19/03/2017 0:00',
    #                                        dataFrame_['WHOIS_UPDATED_DATE'])
    #
    # dataFrame_['WHOIS_REGDATE'] = pd.to_datetime(dataFrame_['WHOIS_REGDATE'])
    # dataFrame_['WHOIS_REGDATE'] = dataFrame_['WHOIS_REGDATE'].map(dt.datetime.toordinal)
    #
    # dataFrame_['WHOIS_UPDATED_DATE'] = pd.to_datetime(dataFrame_['WHOIS_UPDATED_DATE'])
    # dataFrame_['WHOIS_UPDATED_DATE'] = dataFrame_['WHOIS_UPDATED_DATE'].map(dt.datetime.toordinal)
    #
    # print pd.to_datetime('None')

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


    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    print "Training Dataset"
    print "................"
    print "Benign: %d" % sum(float(num) == 0 for num in y_train)
    print "Malicious: %d" % sum(float(num) == 1 for num in y_train)

    print "\nTest Dataset"
    print "................"
    print "Benign: %d" % sum(float(num) == 0 for num in y_test)
    print "Malicious: %d" % sum(float(num) == 1 for num in y_test)

    return x_train, x_test, y_train, y_test

