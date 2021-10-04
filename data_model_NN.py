# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 22:24:44 2021

@author: PatCa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import pickle

import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
import os

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, balanced_accuracy_score,f1_score, \
    precision_score, recall_score, roc_auc_score, classification_report,confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

#Import raw data
source_feature = pd.read_csv('data/features.csv')
source_label = pd.read_csv('data/labels.csv')

#Combine features and labels and copy
source_data = pd.merge(source_feature, source_label, on="trackID")
clean_data = source_data.copy()

#Remove na and duplicates
clean_data = clean_data.dropna()
clean_data = clean_data.drop_duplicates()

#Check type
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

#Remove small categories
clean_data = clean_data[clean_data.time_signature != 0]

#Separate out text feature "tags" and remove from clean_data dataframe
word_df = pd.DataFrame(data=clean_data[['tags','genre']].to_numpy(), columns=['tags','genre'])
clean_data = clean_data.drop(columns=['title'])


#%%Check feature correlation

nc_cols = ['trackID','loudness','tempo','time_signature','key','mode','duration','genre']
tag_col = ['tags']
cat_feat = ['time_signature','key','mode']

cont_data = clean_data.drop(columns=nc_cols+tag_col)

#Remove identical cols
cont_data.loc[:,~cont_data.T.duplicated(keep='first')]
#Get correlation
cont_corr = cont_data.corr()

#IDentify features with more than 85% correlation
corr_list = []
cell_list = []
for j in range(len(cont_corr)):
    for i in range(j,len(cont_corr)):
        if abs(cont_corr.iloc[i,j]) > 0.85:
            if i != j:
                corr_list.append((i+1,j+1))
                cell_list.append(i+1)

cell_set = list(set(cell_list))
vect_cols = ['vect_'+ str(i) for i in cell_set]
cont_data2 = cont_data.drop(columns=vect_cols,axis=1)
cont_corr2 = cont_data2.corr()

#%% Remove filters with more than 85% correlation

vect_cols = ['vect_'+ str(i) for i in cell_set]
clean_data2 = clean_data.drop(columns = vect_cols)
clean_data2.to_csv('data_artifacts/clean_data2.csv', header=False)

#%%Split data for training and testing


train_data = clean_data2
y = train_data[['genre']] #Make Dataframe
training_data = train_data.loc[:,train_data.columns != 'genre']

(X_train, X_test, Y_train, Y_test) = train_test_split(training_data, y,
                                                      test_size=0.2,
                                                      random_state=42,stratify=y)

#Separate out text data
word_df_train = pd.concat((X_train['tags'],Y_train), axis=1) 
word_df_test = pd.concat((X_test['tags'],Y_test), axis=1)
X_train = X_train.drop(columns='tags')
X_test = X_test.drop(columns='tags')

#%% Make pipeline for all data except text data

cat_cols = X_train.select_dtypes("object").columns.to_list()
num_cols = X_train.select_dtypes("number").columns.to_list()
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

#%% Make neural network

WORKING_DIR = os.getcwd() 

#Clean text data
#Remova comma to make a vector of word
word_df['tags'] = word_df['tags'].str.replace(',', '')
word_df_train['tags'] = word_df_train['tags'].str.replace(',','')
word_df_test['tags'] = word_df_test['tags'].str.replace(',','')

def preprocess_data(x):
    #get dataframe column to list
    text = x.to_list()
    #Get right text structure
    text = [str(t).encode('ascii', 'replace') for t in text]
    #Make text np.array for feeding into NN
    text = np.array(text, dtype=object)[:]
    
    return text

text_train = preprocess_data(word_df_train['tags']) 
text_test = preprocess_data(word_df_test['tags']) 

# Make labels one hot for training nn
y_labels = np.array(pd.get_dummies(Y_train['genre'], dtype=int))
y_labels_test = np.array(pd.get_dummies(Y_test['genre'], dtype=int))

#Get input shape for numerical matrix
n_cols = X_train_pipe.shape[1]
#Download lyer Use a pretrained layer from Tensorflow hub to do text embedding
hub_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape=[128], 
                        input_shape=[], dtype=tf.string, name='hub', trainable=False)
#Input layer for embedding layer
text_input = keras.Input(shape=(), name="emb_text_input", dtype=tf.string)
#Embeddinglayer
emb_text = hub_layer(text_input)
#Input 2 numeric matrix
multi_input = keras.Input(shape=(n_cols,), name="multi_data")
#Concatenation layer
x =tf.keras.layers.concatenate([emb_text, multi_input])
#DNN 
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
net = tf.keras.layers.Dense(32, activation='relu')(x)
output = tf.keras.layers.Dense(8, activation='softmax')(net)

multi_model = keras.Model(inputs=[text_input, multi_input],
                       outputs=output)

multi_model.compile(optimizer='adam',
                loss=['categorical_crossentropy'],
                metrics=['accuracy'])

#%% Model description

keras.utils.plot_model(multi_model, to_file="multi_learner_song_predict.png", show_shapes=True)

print(multi_model.summary())


#%%Fit model

history = multi_model.fit({"multi_data":X_train_pipe, "emb_text_input":text_train}, 
                       y_labels, 
                       validation_data=({"multi_data":X_test_pipe, "emb_text_input":text_test}, 
                                        y_labels_test),
                       batch_size=64, epochs=5,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10),
                               tf.keras.callbacks.ModelCheckpoint(os.path.join(WORKING_DIR,
                                                                     'model_checkpoint'),
                                                        monitor='val_loss', verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='auto')])

frame = pd.DataFrame(history.history)

#%% Load best model

multi_model_NN = load_model(os.path.join(WORKING_DIR,'model_checkpoint'))

    
#%% Training plots

plt.figure(dpi=300)
plt.subplot(1,2,1)
plt.plot(frame.accuracy, label='acc')
plt.plot(frame.val_accuracy, color='orange', label='val_Acc')
plt.legend()

plt.subplot(1,2,2)
plt.plot(frame.loss, label='loss')
plt.plot(frame.val_loss, color='orange', label='val_loss')
plt.legend()
plt.show()

#%% Cunfusion matrix

#Predict train
y_pred_multi = multi_model_NN.predict({"multi_data":X_train_pipe, "emb_text_input":text_train} )
y_pred_multi_cons = np.argmax(y_pred_multi, axis=1) + 1
#Predict test
y_pred_test_multi = multi_model_NN.predict({"multi_data":X_test_pipe, "emb_text_input":text_test})
y_pred_test_multi_cons = np.argmax(y_pred_test_multi, axis=1) + 1

# to plot and understand confusion matrix
cm = confusion_matrix(Y_test, y_pred_test_multi_cons)
plot_confusion_matrix(cm)
plt.gcf().set_dpi(300)
plt.show()



#%% Model metriccs


# Model Evaluation
def mod_metrics(Y_test, y_pred_test):
    ac_sc = accuracy_score(Y_test, y_pred_test)
    rc_sc = recall_score(Y_test, y_pred_test, average="weighted")
    pr_sc = precision_score(Y_test, y_pred_test, average="weighted")
    f1_sc = f1_score(Y_test, y_pred_test, average='micro')
    #auc_sc = roc_auc_score(Y_test, y_pred_test,multi_class='ovo')
    
    print('Accuracy: {:.2f}, Precision: {:.2f}, Recall: {:.2f}, F1: {:.2f}'.format(ac_sc, rc_sc, pr_sc, f1_sc))
    #print(classification_report(Y_test, y_pred_test))
    return

mod_metrics(Y_test, y_pred_test_multi_cons)























