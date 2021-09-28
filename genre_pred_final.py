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

#Find duplicates
# a = clean_data.duplicated(subset=None, keep='first')
# count_dup = clean_data[a]

# a = clean_data.T.duplicated(subset=None, keep='first')
# dup_col = clean_data.T[a]

#print(clean_data.isnull().sum().sum())
clean_data = clean_data.dropna()
clean_data = clean_data.drop_duplicates()

#print(clean_data.dtypes)
clean_data = clean_data.astype({'time_signature':int,'key':int,'mode':int})
#print(clean_data.dtypes)

#Rename categorical values
mode_dict = {0:'minor', 1:'major'}
key_dict = {0:'C', 1:'D', 2:'E',3:'F', 4:'G', 5:'H', 6:'I', 7:'J', 8:'K', 9:'L',
            10:'M', 11:'N'}
#ts_dict = {0:'A',1:'B',3:'C',4:'D',5:'E',7:'F'}
label_dict = {'soul and reggae':1, 'pop':2, 'punk':3, 'jazz and blues':4, 
              'dance and electronica':5,'folk':6, 'classic pop and rock':7, 'metal':8}

clean_data['mode'] = clean_data['mode'].replace(mode_dict)
clean_data['key'] = clean_data['key'].replace(key_dict)
#clean_data['time_signature'] = clean_data['time_signature'].replace(ts_dict)
clean_data['genre'] = clean_data['genre'].replace(label_dict)

print(clean_data.iloc[:,0:7].dtypes)


#%% Checking 
nc_cols = ['trackID','loudness','tempo','time_signature','key','mode','duration','genre']
cat_feat = ['time_signature','key','mode']

for i in cat_feat:
    ord_temp = clean_data[i].value_counts()/len(clean_data)
    plt.figure(dpi=300)
    plt.bar(ord_temp.index.astype(str).to_list(),ord_temp)
    plt.title(i)
    plt.show()

clean_data = clean_data[clean_data.time_signature != 0]

#%%Cont features check correlation

cont_data = clean_data.drop(columns=nc_cols)
print(cont_data.shape)

#Remove identical cols
cont_data.loc[:,~cont_data.T.duplicated(keep='first')]
print(cont_data.shape)

cont_corr = cont_data.corr()

#Correalation
plt.figure(dpi=300)
ax = sns.heatmap(cont_corr, annot=False)
plt.show()

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

#Correalation
plt.figure(dpi=300)
ax = sns.heatmap(cont_corr2, annot=False)
plt.show()


#%% Filter

vect_cols = ['vect_'+ str(i) for i in cell_set]
clean_data2 = clean_data.drop(columns = vect_cols)

#%% Check variance


# Import VarianceThreshold from Scikit-learn
from sklearn.feature_selection import VarianceThreshold

feature_names = cont_data2.columns
# Create VarianceThreshold object to perform variance thresholding
selector = VarianceThreshold()

# Perform variance thresholding
selector.fit_transform(cont_data2)

# Print the name and variance of each feature
for feature in zip(feature_names, selector.variances_):
     print(feature)

xy = np.array([feature_names, selector.variances_]).T

xy1 = xy[xy[:,1] < 10000]
xy2 = xy1[xy1[:,1] > 1]

plt.figure(dpi=300)
plt.bar(xy2[:,0],xy2[:,1])
plt.xticks(rotation=-45)
plt.show()

xy3 = xy2[:,0].tolist()

#%% Filter

clean_data3 = clean_data2[nc_cols + xy3]

# Result is worse after training the model
#%%
train_data = clean_data2


y = train_data[['genre']] #Make Dataframe
#y = OneHotEncoder(sparse=False).fit_transform(label_data.values)

training_data = train_data.loc[:,train_data.columns != 'genre']

(X_train, X_test, Y_train, Y_test) = train_test_split(training_data, y,
                                                      test_size=0.2,
                                                      random_state=42,stratify=y)


#%% Over sample training data

from sklearn.utils import resample

#check label balance
plt.figure(dpi=300)
sns.countplot(x ='genre' , data=clean_data)
plt.xticks(rotation=-45, ha='left')
plt.show()

xy_train = pd.concat((X_train, Y_train), axis=1)
largest_genre = xy_train.genre.value_counts().idxmax()
largest_size = xy_train.genre.value_counts().max()

df_1 = xy_train[xy_train.genre == 1]
df_2 = xy_train[xy_train.genre == 2]
df_3 = xy_train[xy_train.genre == 3]
df_4 = xy_train[xy_train.genre == 4]
df_5 = xy_train[xy_train.genre == 5]
df_6 = xy_train[xy_train.genre == 6]
df_7 = xy_train[xy_train.genre == 7]
df_8 = xy_train[xy_train.genre == 8]

df_1_upsampled = resample(df_1, replace=True, n_samples=largest_size, random_state=42)
df_2_upsampled = resample(df_2, replace=True, n_samples=largest_size, random_state=42)
df_3_upsampled = resample(df_3, replace=True, n_samples=largest_size, random_state=42)
df_4_upsampled = resample(df_4, replace=True, n_samples=largest_size, random_state=42)
df_5_upsampled = resample(df_5, replace=True, n_samples=largest_size, random_state=42)
df_6_upsampled = resample(df_6, replace=True, n_samples=largest_size, random_state=42)
df_8_upsampled = resample(df_8, replace=True, n_samples=largest_size, random_state=42)

df_upsampled = pd.concat((df_1_upsampled, df_2_upsampled), axis=0)
df_upsampled = pd.concat((df_upsampled, df_3_upsampled), axis=0)    
df_upsampled = pd.concat((df_upsampled, df_4_upsampled), axis=0)
df_upsampled = pd.concat((df_upsampled, df_5_upsampled), axis=0)
df_upsampled = pd.concat((df_upsampled, df_6_upsampled), axis=0)
df_upsampled = pd.concat((df_upsampled, df_7), axis=0)
df_upsampled = pd.concat((df_upsampled, df_8_upsampled), axis=0)

#check label balance
plt.figure(dpi=300)
sns.countplot(x ='genre' , data=df_upsampled)
plt.xticks(rotation=-45, ha='left')
plt.show()


data_length = df_upsampled.shape[0]
rand_list = np.random.choice(data_length, size=data_length,replace=False).tolist()
df_upsampled = df_upsampled.iloc[rand_list].reset_index(drop=True)
X_train = df_upsampled.iloc[:,:-1]
Y_train = df_upsampled.iloc[:,-1]







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

joblib.dump(pipe, 'C:/Users/PatCa/Documents/PythonScripts/DBS/pipe.joblib')


#%% XGB baseline

# Define model
eval_set = [(X_train_pipe, Y_train),(X_test_pipe, Y_test)]
eval_metric = ['mlogloss']
clf = XGBClassifier()

# Send values to pipeline
xgb_baseline = clf.fit(X_train_pipe, Y_train,
                       eval_set=eval_set, eval_metric=eval_metric,
                       early_stopping_rounds=10, verbose=True)
y_pred = xgb_baseline.predict(X_train_pipe)
y_pred_test = xgb_baseline.predict(X_test_pipe)
feature_importance = clf.feature_importances_

#%%
from multiprocessing import cpu_count
import datetime


eval_metric = ['mlogloss']
#Params
XGB_Options = { 
                'n_jobs':           cpu_count()//2,
                'cv':               3,
                'scoring':          'accuracy', #'f1_macro','roc_auc_ovr',
                'seed':             42, 
    
                'max_depth':        [6,9,12],
                'min_child_weight': [1,4,7],
                'n_estimators':     [15], 

                'gamma':            [0, 0.2, 0.4],
                'colsample_bytree': [0.9, 1.0, 1.1],
                'subsample':        [0.9, 1.0, 1.1],
                'reg_alpha':        [1e-6, 1, 10], 
                'reg_lambda':       [1e-6, 1, 10],
                'learning_rate':    [0.2, 0.3, 0.4],
                 'verbose':         1            
              }#

xgb_reg = xgb.XGBClassifier(random_state = XGB_Options['seed'],
                             n_jobs = cpu_count()//2,
                             eval_metric = eval_metric, 
                             verbosity=0)
start = datetime.datetime.now()
print('Starting with low learning rate and tuning: max_depth, min_child_weight, n_estimators')

params = {  
    "learning_rate":     [0.1],
    "max_depth":         XGB_Options['max_depth'], 
    "min_child_weight":  XGB_Options['min_child_weight'], 
    "n_estimators":      XGB_Options['n_estimators'], 

    "colsample_bytree":  [0.9], 
    "subsample":         [0.9],
    "gamma":             [0],
}

GSCV = GridSearchCV(xgb_reg, 
                    params,
                    cv                 = XGB_Options['cv'],
                    scoring            = XGB_Options['scoring'], 
                    n_jobs             = XGB_Options['n_jobs'], 
                    verbose            = XGB_Options['verbose'], 
                    return_train_score = True)

GSCV.fit(X_train_pipe, Y_train.values)
end = datetime.datetime.now()

print('Time to fit', (end-start))
print('best_params_:', GSCV.best_params_)
print('best_score_:',  GSCV.best_score_)

print('Tuning: gamma')
start = datetime.datetime.now()
params = {  
    "learning_rate":    [0.1], 
    "max_depth":        [GSCV.best_params_['max_depth']],
    "min_child_weight": [GSCV.best_params_['min_child_weight']],
    "n_estimators":     [GSCV.best_params_['n_estimators']],

    "colsample_bytree": [0.9], 
    "subsample":        [0.9],
    "gamma":            XGB_Options['gamma'],

}

GSCV = GridSearchCV(xgb_reg, 
                    params,
                    cv                 = XGB_Options['cv'],
                    scoring            = XGB_Options['scoring'], 
                    n_jobs             = XGB_Options['n_jobs'],
                    verbose            = XGB_Options['verbose'], 
                    return_train_score = True)

GSCV.fit(X_train_pipe, Y_train.values)
end = datetime.datetime.now()

print('Time to fit xgb', (end-start))
print('best_params_:', GSCV.best_params_)#, 
print('best_score_:', GSCV.best_score_)


print('Tuning: colsample_bytree, subsample')
start = datetime.datetime.now()

params = {  
    "learning_rate":    [0.1], 
    "max_depth":        [GSCV.best_params_['max_depth']],
    "min_child_weight": [GSCV.best_params_['min_child_weight']],
    "n_estimators":     [GSCV.best_params_['n_estimators']],
    "gamma":            [GSCV.best_params_['gamma']],

    "colsample_bytree": XGB_Options['colsample_bytree'],
    "subsample":        XGB_Options['subsample'],

}

GSCV = GridSearchCV(xgb_reg, 
                    params,
                    cv                 = XGB_Options['cv'],
                    scoring            = XGB_Options['scoring'], 
                    n_jobs             = XGB_Options['n_jobs'],
                    verbose            = XGB_Options['verbose'], 
                    return_train_score = True)

GSCV.fit(X_train_pipe, Y_train.values)
end = datetime.datetime.now()

print('Time to fit', (end-start))
print('best_params_:', GSCV.best_params_) 
print('best_score_:', GSCV.best_score_)

print('Tuning: reg_alpha, reg_lambda')
start = datetime.datetime.now()
params = {  
    "learning_rate":    [0.1], 
    "max_depth":        [GSCV.best_params_['max_depth']],
    "min_child_weight": [GSCV.best_params_['min_child_weight']],
    "n_estimators":     [GSCV.best_params_['n_estimators']],
    "gamma":            [GSCV.best_params_['gamma']],

    "colsample_bytree": [GSCV.best_params_['colsample_bytree']], 
    "subsample":        [GSCV.best_params_['subsample']],


    "reg_alpha":        XGB_Options['reg_alpha'], 
    "reg_lambda":       XGB_Options['reg_lambda'], 
}

GSCV = GridSearchCV(xgb_reg, 
                    params,
                    cv                 = XGB_Options['cv'],
                    scoring            = XGB_Options['scoring'], 
                    n_jobs             = XGB_Options['n_jobs'],
                    verbose            = XGB_Options['verbose'], 
                    return_train_score = True)

GSCV.fit(X_train_pipe, Y_train.values)
end = datetime.datetime.now()

print('Time to fit', (end-start))
print('best_params_:', GSCV.best_params_)
print('best_score_:', GSCV.best_score_)


print('Tuning: learning_rate')
start = datetime.datetime.now()

params = {  
    "learning_rate":    XGB_Options['learning_rate'], 
    "max_depth":        [GSCV.best_params_['max_depth']],
    "min_child_weight": [GSCV.best_params_['min_child_weight']],
    "n_estimators":     [GSCV.best_params_['n_estimators']],
    "gamma":            [GSCV.best_params_['gamma']],

    "colsample_bytree": [GSCV.best_params_['colsample_bytree']], 
    "subsample":        [GSCV.best_params_['subsample']],


    "reg_alpha":        [GSCV.best_params_['reg_alpha']],
    "reg_lambda":       [GSCV.best_params_['reg_lambda']]
}

GSCV = GridSearchCV(xgb_reg, 
                    params,
                    cv                 = XGB_Options['cv'],
                    scoring            = XGB_Options['scoring'], 
                    n_jobs             = XGB_Options['n_jobs'],
                    verbose            = XGB_Options['verbose'], 
                    return_train_score = True)

GSCV.fit(X_train_pipe, Y_train.values) 
end = datetime.datetime.now()

print('Time to fit', (end-start))
print('best_params_:', GSCV.best_params_)
print('best_score_:', GSCV.best_score_)

#%% Train tuned
# Define model
eval_set = [(X_train_pipe, Y_train),(X_test_pipe, Y_test)]
eval_metric = ['mlogloss']
start = datetime.datetime.now()
xgb_tuned           = xgb.XGBClassifier(random_state=XGB_Options['seed']\
                                        , n_jobs=cpu_count()//2) #seed)
xgb_tuned.set_params(**GSCV.best_params_)
trained_xgb_tuned   = xgb_tuned.fit(X_train_pipe, Y_train,
                                    eval_metric = eval_metric,
                                    eval_set = eval_set, verbose=False,
                                    early_stopping_rounds=10)
feature_importances = trained_xgb_tuned.feature_importances_ 
end = datetime.datetime.now()
print('Time to fit xgb', (end-start))

y_pred = trained_xgb_tuned.predict(X_train_pipe)
y_pred_test = trained_xgb_tuned.predict(X_test_pipe)
feature_importance = xgb_tuned.feature_importances_


#%% Confusion matrix

# to plot and understand confusion matrix
cm = confusion_matrix(Y_test, y_pred_test)
plot_confusion_matrix(cm)
plt.gcf().set_dpi(300)
plt.show()

#%%
# Model Evaluation
ac_sc = accuracy_score(Y_test, y_pred_test)
rc_sc = recall_score(Y_test, y_pred_test, average="weighted")
pr_sc = precision_score(Y_test, y_pred_test, average="weighted")
f1_sc = f1_score(Y_test, y_pred_test, average='micro')
#auc_sc = roc_auc_score(Y_test, y_pred_test,multi_class='ovo')

print('Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(ac_sc, rc_sc, pr_sc, f1_sc))
print(classification_report(Y_test, y_pred_test))

#%% Save and load trained model

file_name = "c:/Users/PatCa/Documents/PythonScripts/DBS/xgb_baseline.pkl"

pickle.dump(xgb_baseline, open(file_name,'wb'))

imported_xgb_baseline = pickle.load(open(file_name,'rb'))

#%% Test predict

sample_row = 99

test_data = pd.read_csv('C:/Users/PatCa/Documents/PythonScripts/DBS/data/test.csv')
test_data = test_data.iloc[sample_row,:].to_frame().T

test_data = test_data.copy().drop(columns=['title', 'tags'])

test_data = test_data.astype({'time_signature':int,'key':int,'mode':int})
#print(clean_data.dtypes)

#Rename categorical values
mode_dict = {0:'minor', 1:'major'}
key_dict = {0:'C', 1:'D', 2:'E',3:'F', 4:'G', 5:'H', 6:'I', 7:'J', 8:'K', 9:'L',
            10:'M', 11:'N'}
#ts_dict = {0:'A',1:'B',3:'C',4:'D',5:'E',7:'F'}
test_data['mode'] = test_data['mode'].replace(mode_dict)
test_data['key'] = test_data['key'].replace(key_dict)
#clean_data['time_signature'] = clean_data['time_signature'].replace(ts_dict)
test_data = test_data.drop(columns = vect_cols)
test_pipe = pipe.transform(test_data)
y_pred = clf.predict(test_pipe)


rev_label_dict = {1:'soul and reggae', 2:'pop', 3:'punk', 4:'jazz and blues', 
              5:'dance and electronica',6:'folk', 7:'classic pop and rock', 8:'metal'}
print(rev_label_dict[y_pred[0]])











