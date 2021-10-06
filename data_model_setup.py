# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:24:44 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import pickle

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

source_feature = pd.read_csv('data/features.csv')
source_label = pd.read_csv('data/labels.csv')

result = pd.merge(source_feature, source_label, on="trackID")
clean_data = result.copy().drop(columns=['title', 'tags', 'trackID'])

clean_data = clean_data.dropna()
clean_data = clean_data.drop_duplicates()

clean_data = clean_data.astype({'time_signature':int,'key':int,'mode':int})
#Rename categorical values
mode_dict = {0:'minor', 1:'major'}
key_dict = {0:'C', 1:'D', 2:'E',3:'F', 4:'G', 5:'H', 6:'I', 7:'J', 8:'K', 9:'L',
            10:'M', 11:'N'}
label_dict = {'soul and reggae':1, 'pop':2, 'punk':3, 'jazz and blues':4, 
              'dance and electronica':5,'folk':6, 'classic pop and rock':7, 'metal':8}

clean_data['mode'] = clean_data['mode'].replace(mode_dict)
clean_data['key'] = clean_data['key'].replace(key_dict)
clean_data['genre'] = clean_data['genre'].replace(label_dict)

#Remove category 0
clean_data = clean_data[clean_data.time_signature != 0]
#Remove identical cols
clean_data.loc[:,~clean_data.T.duplicated(keep='first')]


#%%
train_data = clean_data
y = train_data[['genre']] #Make Dataframe
training_data = train_data.loc[:,train_data.columns != 'genre']

(X_train, X_test, Y_train, Y_test) = train_test_split(training_data, y,
                                                      test_size=0.2,
                                                      random_state=42,stratify=y)
#%%Cont features check correlation
nc_cols = ['loudness','tempo','duration']
cat_cols = ['time_signature','key','mode']

#Get continuous data
cont_data = X_train.drop(columns=nc_cols+cat_cols)
#Get correlation
cont_corr = cont_data.corr()

#Remove correlations larger than 0.85
corr_list = []
cell_list = []
for j in range(len(cont_corr)):
    for i in range(j,len(cont_corr)):
        if abs(cont_corr.iloc[i,j]) > 0.85:
            if i != j:
                corr_list.append((i+1,j+1))
                cell_list.append(i+1)

cell_set = list(set(cell_list))
vect_cols_to_remove = ['vect_'+ str(i) for i in cell_set]
all_vect_cols = ['vect_' + str(i+1) for i in range(148)]
vect_cols_to_keep = [i for i in all_vect_cols if i not in vect_cols_to_remove]

#%% Make pipeline
num_cols = nc_cols + vect_cols_to_keep
feature_ar = np.array(num_cols + cat_cols)

# Apply preprocess
preprocessor = ColumnTransformer(
    transformers=[('nums', RobustScaler(), num_cols), 
                  ('cats', OneHotEncoder(sparse=False), cat_cols)],
                  remainder='drop')

# ('pass','passthrough', num_cols)

#Transform data
pipe = Pipeline(steps=[('preprocessor',preprocessor)])
X_train_pipe = pipe.fit_transform(X_train)
X_test_pipe = pipe.transform(X_test)
#print(pipe.steps)

#Make list of feature names after one-hot
feature_list = []
for name, estimator, features in preprocessor.transformers_:
    if hasattr(estimator,'get_feature_names'):
        if isinstance(estimator,OneHotEncoder):
            f = estimator.get_feature_names(features)
            feature_list.extend(f)
    else:
        feature_list.extend(features)

joblib.dump(pipe, 'model_artifacts/pipe.joblib')


#%% XGB baseline

# Define model
eval_set = [(X_train_pipe,Y_train), (X_test_pipe, Y_test)]
eval_metric = ['mlogloss']
clf = XGBClassifier()

# Send values to pipeline
xgb_baseline = clf.fit(X_train_pipe,Y_train,
                       eval_set=eval_set, eval_metric=eval_metric,
                       early_stopping_rounds=10, verbose=True)

#%% Save and load trained model

file_name = "model_artifacts/xgb_baseline.pkl"

pickle.dump(xgb_baseline, open(file_name,'wb'))












