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
clean_data = clean_data.drop(columns=['title','trackID'])

#%%Split data for training and testing

train_data = clean_data
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

#%%Check feature correlation

nc_cols = ['loudness','tempo','time_signature','key','mode','duration']
cat_feat = ['time_signature','key','mode']

cont_data = X_train.drop(columns=nc_cols)

#%% PCA on cont_data2
from sklearn.decomposition import PCA

cont_data_norm = (cont_data-cont_data.mean())/(cont_data.std())

pca = PCA()
principalcomponents = pca.fit_transform(cont_data_norm)

cont_data_pca = pd.DataFrame(data = principalcomponents)

#%%
comp_var = pca.explained_variance_ratio_
xx = [i for i in range(len(comp_var))]

plt.figure()
plt.subplot(1,2,1)
plt.bar(xx,comp_var)
plt.grid()
plt.subplot(1,2,2)
plt.plot(np.cumsum(comp_var))
plt.grid()
plt.show()

print("30 columns explain {}% of the variation".format(round(np.sum(comp_var[:30]),2)))

#%%

cont_data_norm = (cont_data-cont_data.mean())/(cont_data.std())

pca = PCA(0.95).fit(cont_data_norm)
num_pca_cols = pca.n_components_
print(num_pca_cols)

data_pca_array = pca.transform(cont_data_norm)
cont_data_pca = pd.DataFrame(data=data_pca_array)
col_names = ['PCA_'+str(i) for i in range(num_pca_cols)]
cont_data_pca.columns = col_names

X_train2 = X_train[nc_cols]
X_train3 = pd.concat([X_train2.reset_index(drop=True), cont_data_pca.reset_index(drop=True)], axis=1)

#%% Transform test data

cont_test_data = X_test.drop(columns=nc_cols)
cont_test_data_norm = (cont_test_data-cont_test_data.mean())/(cont_test_data.std())

test_data_pca_array = pca.transform(cont_test_data_norm)
cont_test_data_pca = pd.DataFrame(data=test_data_pca_array)
test_col_names = ['PCA_'+str(i) for i in range(num_pca_cols)]
cont_test_data_pca.columns = test_col_names

X_test2 = X_test[nc_cols]
X_test3 = pd.concat((X_test2.reset_index(drop=True), cont_test_data_pca.reset_index(drop=True)), axis=1)

#%% Make pipeline for all data except text data

cat_cols = ['time_signature','key','mode']
num_cols = ['loudness','tempo','duration']
pca_cols = col_names
label_cols = ['genre']
feature_ar = np.array(num_cols + cat_cols + pca_cols)

# Apply preprocess
preprocessor = ColumnTransformer(
    transformers=[('nums', RobustScaler(), num_cols), 
                  ('cats', OneHotEncoder(sparse=False), cat_cols),
                  ('pca_cols','passthrough', pca_cols)])

# ('pass','passthrough', num_cols)

#Transform data
pipe = Pipeline(steps=[('preprocessor',preprocessor)])
X_train_pipe = pipe.fit_transform(X_train3)
X_test_pipe = pipe.transform(X_test3)
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
eval_set = [(X_train_pipe,Y_train),(X_test_pipe,Y_test)]
eval_metric = ['mlogloss']
clf = XGBClassifier()

# Send values to pipeline
xgb_baseline1 = clf.fit(X_train_pipe,Y_train,
                       eval_set=eval_set, eval_metric=eval_metric,
                       early_stopping_rounds=10, verbose=True)

xgb_pred = xgb_baseline1.predict(X_train_pipe)
xgb_pred_test = xgb_baseline1.predict(X_test_pipe)

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

mod_metrics(Y_test, xgb_pred_test)

# to plot and understand confusion matrix
cm = confusion_matrix(Y_test, xgb_pred_test)
plot_confusion_matrix(cm)
plt.gcf().set_dpi(300)
plt.show()

#%% Save and load trained model

file_name = "model_artifacts/xgb_baseline1.pkl"

pickle.dump(xgb_baseline1, open(file_name,'wb'))

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
x = tf.keras.layers.Dense(n_cols, activation='relu')(x)
x = tf.keras.layers.Dense(n_cols, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(n_cols, activation='relu')(x)
x = tf.keras.layers.Dense(n_cols, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
net = tf.keras.layers.Dense(32, activation='relu')(x)
output = tf.keras.layers.Dense(8, activation='softmax')(net)

multi_model = keras.Model(inputs=[text_input, multi_input],
                       outputs=output)

multi_model.compile(optimizer='adam',
                loss=['categorical_crossentropy'],
                metrics=['accuracy'])

#%% Model description

keras.utils.plot_model(multi_model, to_file="multi_learner_song_predict.png", show_shapes=False)

print(multi_model.summary())


#%%Fit model

history = multi_model.fit({"multi_data":X_train_pipe, "emb_text_input":text_train}, 
                       y_labels, 
                       validation_data=({"multi_data":X_test_pipe, "emb_text_input":text_test}, 
                                        y_labels_test),
                       batch_size=64, epochs=100,
                    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5),
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























