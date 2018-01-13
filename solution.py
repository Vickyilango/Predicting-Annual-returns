# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 12:18:40 2017

@author: VIGNESHWAR
"""

import numpy as np
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#data preprocessing
sub_ids = test['portfolio_id']
test = test.drop('portfolio_id', axis=1)
y = train['return']
train = train.drop(['portfolio_id', 'return'], axis=1)

train = train.drop(['start_date', 'creation_date', 'sell_date', 'indicator_code', 'status', 'desk_id'], axis=1)
test = test.drop(['start_date', 'creation_date', 'sell_date', 'indicator_code', 'status', 'desk_id'], axis=1)


#filling with true ort false
train['hedge_value'].fillna(False, inplace=True)
test['hedge_value'].fillna(False, inplace=True)
#filling missing numbers with median
train['sold'].fillna(train['sold'].median(), inplace=True)
train['bought'].fillna(train['bought'].median(), inplace=True)
train['libor_rate'].fillna(train['libor_rate'].median(), inplace=True)
test['libor_rate'].fillna(train['libor_rate'].median(), inplace=True)

#enocoding 
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
obj_cols = [x for x in train.columns if train[x].dtype == 'object']
encoder = LabelEncoder()
for x in obj_cols:
    encoder.fit(train[x])
    train[x] = encoder.transform(train[x])
    test[x] = encoder.transform(test[x])
labelencoder= LabelEncoder()
train['hedge_value'] = labelencoder.fit_transform(train['hedge_value'])
test['hedge_value'] = labelencoder.fit_transform(test['hedge_value'])  

#concatting for using onehotencoder
train_objs_num = len(train)
dataset = pd.concat(objs=[train, test], axis=0)
dataset = pd.get_dummies(dataset)
#using onehotencoder
onehotencoder = OneHotEncoder(categorical_features=[1,3,5,8]) 
dataset = onehotencoder.fit_transform(dataset).toarray() 
#briningback the datas
train = dataset[:train_objs_num]
test = dataset[train_objs_num:]


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
train = sc_x.fit_transform(train)
test = sc_x.fit_transform(test)
   

#forestrefressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=7)
scores = cross_val_score(forest_reg, train, y, scoring='r2', cv=5)
print(scores)
print('mean r2:',np.mean(scores))


from IPython.display import FileLink

forest_reg = RandomForestRegressor(random_state=7)
forest_reg.fit(train, y)
preds = forest_reg.predict(test)

sub = pd.DataFrame({'portfolio_id': sub_ids, 'return': preds})
filename = 'solution.csv'
sub.to_csv(filename, index=False)
FileLink(filename) 
