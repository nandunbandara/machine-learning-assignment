import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from dateutil.parser import parse


np.set_printoptions(threshold=np.nan)


# check if a passed string in datetime format
def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False


def preprocess_training_data(path):

    # Import dataset
    dataframe = pd.read_csv(path)

    include = ['URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS',
               'CONTENT_LENGTH','WHOIS_COUNTRY','WHOIS_STATEPRO',
               'WHOIS_REGDATE','WHOIS_UPDATED_DATE','TCP_CONVERSATION_EXCHANGE',
               'DIST_REMOTE_TCP_PORT','REMOTE_IPS','APP_BYTES','SOURCE_APP_PACKETS',
               'REMOTE_APP_PACKETS','APP_PACKETS','DNS_QUERY_TIMES','Type']

    dataframe_ = dataframe[include]

    # remove invalid datetime records
    for i,v in dataframe_['WHOIS_REGDATE'].items():
        if not is_date(v):
            dataframe_.loc[i,['WHOIS_REGDATE']] = '19/03/2017 0:00'

    for i, v in dataframe_['WHOIS_UPDATED_DATE'].items():
        if not is_date(v):
            dataframe_.loc[i,['WHOIS_UPDATED_DATE']] = '19/03/2017 0:00'

    try:

        # convert datetime records to timestamps
        dataframe_.loc[:,'WHOIS_REGDATE'] = pd.to_datetime(dataframe_['WHOIS_REGDATE'])
        dataframe_.loc[:,'WHOIS_REGDATE'] = dataframe_['WHOIS_REGDATE'].map(dt.datetime.toordinal)

        dataframe_.loc[:,'WHOIS_UPDATED_DATE'] = pd.to_datetime(dataframe_['WHOIS_UPDATED_DATE'])
        dataframe_.loc[:,'WHOIS_UPDATED_DATE'] = dataframe_['WHOIS_UPDATED_DATE'].map(dt.datetime.toordinal)

    except ValueError:
        print "ERROR: Invalid Date and Time"

    categorical_columns = []
    for col, col_type in dataframe_.dtypes.iteritems():
        if col_type == 'O':
            categorical_columns.append(col)

    dataframe_encoded = pd.get_dummies(dataframe_, columns=categorical_columns, dummy_na=True)

    # handle null values
    # set the mean value of the column for nan
    impute = Imputer(missing_values='NaN', strategy='mean', axis=0)
    impute = impute.fit(dataframe_encoded)
    dataframe_encoded[:] = impute.transform(dataframe_encoded[:])

    # separate dependent and independent variables
    dependent_variable = 'Type'

    x = dataframe_encoded[dataframe_encoded.columns.difference([dependent_variable])]
    y = dataframe_encoded[dependent_variable]

    # standardize the data set
    scale = MinMaxScaler()
    x = scale.fit_transform(x)

    # split into train and test data sets
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

