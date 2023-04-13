## IMPORT DEPENDENCIES ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

## DATA IMPORT ## 
## For this project the data from https://www.kaggle.com/c/ieee-fraud-detection/data has been used. ##

data_path = "./data/"

train_tr = pd.read_csv(data_path + "train_transaction.csv")
train_id = pd.read_csv(data_path + "train_identity.csv") 
test_tr = pd.read_csv(data_path + "test_transaction.csv")
test_id = pd.read_csv(data_path + "test_identity.csv")

print('train_transaction shape is: {}'.format(train_tr.shape))
print('train_identity shape is: {}'.format(train_id.shape))

print('test_transaction shape is: {}'.format(test_tr.shape))
print('test_identity shape is: {}'.format(test_id.shape))


## DATA MANIPULATION PHASE ## 

train = pd.merge(train_tr, train_id, how = 'left', on = 'TransactionID')
test = pd.merge(test_tr, test_id, how = 'left', on = 'TransactionID')
del train_tr, train_id, test_tr, test_id
print('train set shape is: {}'.format(train.shape))
print('test set shape is: {}'.format(test.shape))

def different_columns(traincols, testcols):
    diff_cols = []
    for i in traincols:
        if i not in testcols:
            diff_cols.append(i)
    return diff_cols

test = test.rename(columns = {"id-01": "id_01", "id-02": "id_02", "id-03": "id_03", 
                            "id-06": "id_06", "id-05": "id_05", "id-04": "id_04", 
                            "id-07": "id_07", "id-08": "id_08", "id-09": "id_09", 
                            "id-10": "id_10", "id-11": "id_11", "id-12": "id_12", 
                            "id-15": "id_15", "id-14": "id_14", "id-13": "id_13", 
                            "id-16": "id_16", "id-17": "id_17", "id-18": "id_18", 
                            "id-21": "id_21", "id-20": "id_20", "id-19": "id_19", 
                            "id-22": "id_22", "id-23": "id_23", "id-24": "id_24", 
                            "id-27": "id_27", "id-26": "id_26", "id-25": "id_25", 
                            "id-28": "id_28", "id-29": "id_29", "id-30": "id_30", 
                            "id-31": "id_31", "id-32": "id_32", "id-33": "id_33", 
                            "id-34": "id_34", "id-35": "id_35", "id-36": "id_36", 
                            "id-37": "id_37", "id-38": "id_38"})


## ENCODING VARIABLES

from sklearn import preprocessing
encoder_dict = {}

complete_labelset_temp = pd.concat([train.drop(['isFraud'], axis=1), test], axis=0).reset_index()
variables_encode = complete_labelset_temp.keys()
for k in variables_encode:
    if complete_labelset_temp[k].dtype == object:
        le = preprocessing.LabelEncoder()
        le_fit = le.fit(complete_labelset_temp[k])
        encoder_dict.update({k: le_fit})
        #train[k + '_encoded'] = le_fit.transform(train[k])  
        train[k + '_encoded'] = encoder_dict[k].transform(train[k])  
        train = train.drop([k], axis=1)
        test[k + '_encoded'] = encoder_dict[k].transform(test[k])  
        test = test.drop([k], axis=1)

train.to_csv('./data/train_processed.csv')
test.to_csv('./data/test_processed.csv')

