# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:24:44 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
import pickle

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, balanced_accuracy_score,f1_score, \
    precision_score, recall_score, roc_auc_score, classification_report,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

source_feature = pd.read_csv('data/features.csv')
source_label = pd.read_csv('data/labels.csv')


result = pd.merge(source_feature, source_label, on="trackID")
clean_data = result.copy().drop(columns=['title', 'tags'])


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

clean_data = clean_data[clean_data.time_signature != 0]

#%%Cont features check correlation
nc_cols = ['trackID','loudness','tempo','time_signature','key','mode','duration','genre']
cat_feat = ['time_signature','key','mode']

cont_data = clean_data.drop(columns=nc_cols)

#Remove identical cols
cont_data.loc[:,~cont_data.T.duplicated(keep='first')]
cont_corr = cont_data.corr()


corr_list = []
cell_list = []
for j in range(len(cont_corr)):
    for i in range(j,len(cont_corr)):
        if cont_corr.iloc[i,j] > 0.85:
            if i != j:
                corr_list.append((i+1,j+1))
                cell_list.append(i+1)

cell_set = list(set(cell_list))
vect_cols = ['vect_'+ str(i) for i in cell_set]
cont_data2 = cont_data.drop(columns=vect_cols,axis=1)
cont_corr2 = cont_data2.corr()

#%% Filter

vect_cols = ['vect_'+ str(i) for i in cell_set]
clean_data2 = clean_data.drop(columns = vect_cols)
clean_data2.to_csv('data_artifacts/clean_data2.csv', header=False)

#%%
train_data = clean_data2
y = train_data[['genre']] #Make Dataframe
training_data = train_data.loc[:,train_data.columns != 'genre']

(X_train, X_test, Y_train, Y_test) = train_test_split(training_data, y,
                                                      test_size=0.001,
                                                      random_state=42,stratify=y)

#%% Make pipeline
cat_cols = training_data.select_dtypes("object").columns.to_list()
num_cols = training_data.select_dtypes("number").columns.to_list()
ord_cols = [] #Add beats per miute as ordinal
label_cols = ['genre']
feature_ar = np.array(num_cols + cat_cols)

# Apply preprocess
preprocessor = ColumnTransformer(
    transformers=[('nums', RobustScaler(), num_cols), 
                  ('cats', OneHotEncoder(sparse=False), cat_cols)])

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
eval_set = [(X_train_pipe,Y_train)]
eval_metric = ['mlogloss']
clf = XGBClassifier()

# Send values to pipeline
xgb_baseline = clf.fit(X_train_pipe,Y_train,
                       eval_set=eval_set, eval_metric=eval_metric,
                       early_stopping_rounds=10, verbose=True)

#%% Save and load trained model

file_name = "model_artifacts/xgb_baseline.pkl"

pickle.dump(xgb_baseline, open(file_name,'wb'))












