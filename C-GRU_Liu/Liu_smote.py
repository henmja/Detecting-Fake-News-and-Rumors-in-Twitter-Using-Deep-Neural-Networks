#coding: utf-8
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from langdetect import detect
from  tqdm import tqdm
import nltk
nltk.download('stopwords')
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.convolutional import MaxPooling2D
import nltk
import re
from tqdm import tqdm_notebook
from nltk.corpus import stopwords
from tensorflow import keras
from tensorflow.keras import regularizers, initializers, optimizers, callbacks 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.callbacks import EarlyStopping
from nltk.stem.snowball import SnowballStemmer
from textblob import TextBlob
from keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Input, Dense, Embedding, Dropout, GRU, Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np

df = pd.read_pickle("bigdata_timeseries.pkl")
print('before')

df['created_at'] = df['created_at'].astype(str)
df['followers'] = df['followers'].astype(str)
df['following'] = df['following'].astype(str)

label = pd.get_dummies(df['label'])
label = np.array(label)

num_Cols = ['favorite_count', 'favourites_count_user', 'followers_count_user', 'friends_count_user', 'listed_count_user', 'retweet_count', 'statuses_count_user', 'zero_indices_zero_urls_entities', 'one_indices_zero_urls_entities', 'zero_indices_zero_urls_url_entities_user', 'one_indices_zero_urls_url_entities_user', 'zero_indices_zero_user_mentions_entities', 'one_indices_zero_user_mentions_entities']
num_Features = []
for idx in num_Cols:
    print(idx)
    num_Features.append(df[idx])
print(type(num_Features[0]))
import pandas as pd	
from sklearn import preprocessing

for i,col in enumerate(num_Features):
    if i == 0:	
        mat = col	
    else:
        mat = pd.concat([mat, col], axis=1)
x = mat.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-0.5,0.5))
x_scaled = min_max_scaler.fit_transform(x)
num_Norm = pd.DataFrame(x_scaled)

for col in num_Norm.keys():
    num_Norm[col] = pd.cut(num_Norm[col], bins=3, labels=np.arange(3), right=False)

indices = np.arange(num_Norm.shape[0])
np.random.shuffle(indices)
time_series = df['created_at_retweets']
time_series.reset_index(drop=True, inplace=True)

time_series = time_series[indices]
labels = label[indices]



test_Perc = 0.2 #20% data used for testing
num_validation_samples = int(test_Perc*num_Norm.shape[0])
time_series_Train = time_series[: -num_validation_samples]
num_Norm_Train = num_Norm[: -num_validation_samples]
target_Train = labels[: -num_validation_samples]
num_Norm_Val = num_Norm[-num_validation_samples: ]
num_Norm_Val = num_Norm_Val.to_numpy()
print('NUM NORM VAL')
print(num_Norm_Val)
time_series_Val = time_series[-num_validation_samples: ]
#time_series_Val = time_series_Val.to_numpy()
print(time_series_Train.shape)
print('TIME SERIES VAL')
print(time_series_Val)
print(time_series_Val.shape)
target_Val = labels[-num_validation_samples: ]
print('Number of records in each attribute:')
print('training: ', target_Train.sum(axis=0))
print('validation: ', target_Val.sum(axis=0))

maxLen = 0
maxIndex = 0
time_series_Train = np.asarray(time_series_Train)
time_series_Val = np.asarray(time_series_Val)
for i,times in enumerate(time_series_Train):
    time_series_Train[i] = np.array(time_series_Train[i])
print(time_series_Train.shape)
print(time_series_Val.shape)
for i,times in enumerate(time_series_Val):
    time_series_Val[i] = np.array(time_series_Val[i])
print(time_series_Train.shape)
print(time_series_Val.shape)
max_len = max([len(x) for x in time_series_Train])

for i,times in enumerate(time_series_Train):
	
    time_series_Train[i] = np.pad(times,(0,max_len-len(times)), 'constant')

time_series_Mat = np.zeros((len(time_series_Train),max_len))
	
for i, times in enumerate(time_series_Train):
	
    for j, time in enumerate(time_series_Train[i]):
        time_series_Mat[i,j] = time

for i,times in enumerate(time_series_Val):

    time_series_Val[i] = np.pad(times,(0,max_len-len(times)), 'constant')

time_series_Mat_Val = np.zeros((len(time_series_Val),max_len))

for i, times in enumerate(time_series_Val):

    for j, time in enumerate(time_series_Val[i]):
        time_series_Mat_Val[i,j] = time


from imblearn.over_sampling import SMOTE
oversample = SMOTE()
print(target_Train)
print(target_Train.shape)
print(type(target_Train))
num_Norm_Train = num_Norm_Train.to_numpy()
time_series_Mat, target_Train_Resampled = oversample.fit_resample(time_series_Mat, target_Train)
num_Norm_Train, target_Train = oversample.fit_resample(num_Norm_Train, target_Train)
from keras.utils import to_categorical
target_Train = to_categorical(target_Train)



from keras.layers import Input, Embedding, LSTM, Dense, merge, concatenate, GRU, Conv1D, Flatten, Dropout
from keras.models import Model
from keras import metrics
main_input = Input(shape=(max_len,), dtype='int32', name='main_input')
x = Embedding(output_dim=512, input_dim=10000, input_length=max_len)(main_input)
gru_out = GRU(32)(x)
gru_out = Dropout(0.5)(gru_out)
auxiliary_output = Dense(2, activation='sigmoid', name='aux_output')(gru_out)

auxiliary_input = Input(shape=(13,), name='aux_input')
e = Embedding(output_dim=512, input_dim=10000, input_length=13)(auxiliary_input)
cnn_out = Conv1D(32,3)(e)
cnn_out = Dropout(0.5)(cnn_out)
cnn_out = Flatten()(cnn_out)
x = concatenate([gru_out, cnn_out])
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(2, activation='sigmoid', name='main_output')(x)

model = Model(input=[main_input, auxiliary_input], output=[main_output, auxiliary_output])

model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[1., 0.2],metrics=[metrics.AUC()])
print(time_series_Mat.shape)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.fit([time_series_Mat,num_Norm_Train], [target_Train, target_Train], nb_epoch=2, batch_size=32, callbacks=[es])

print(time_series_Mat_Val.shape)
print(num_Norm_Val.shape)
print(type(time_series_Mat_Val))
print(type(num_Norm_Val))
predictions = model.predict([time_series_Mat_Val,num_Norm_Val])
from sklearn.metrics import classification_report
predictions_bool = np.argmax(predictions[0], axis=1)
import sklearn.metrics as metrics
print(len(predictions))
predictions = np.asarray(predictions)
y_pred = (predictions > 0.5)
print('target_Val shape')
print(target_Val.shape)
print('y_pred shape')
print(y_pred.shape)
print('predictions shape')
print(predictions.shape)
print(type(y_pred[0]))
print(type(y_pred[1]))
y_predictions = y_pred[0]+y_pred[1]
matrix = metrics.confusion_matrix(target_Val.argmax(axis=1), predictions[0].argmax(axis=1))
print('predictions')
print(predictions[0])
print(predictions[0].argmax(axis=1))
print('y_pred')
print(y_pred[0])
print(y_pred[0].argmax(axis=1))
print(matrix)
micro = (matrix[0,0]+matrix[1,1])/(matrix[0,0]+matrix[1,1]+matrix[0,1]+matrix[1,0])
print('micro')
print(micro)
print('macro')
macro = (matrix[0,0]/(matrix[0,0]+matrix[1,0])+matrix[1,1]/(matrix[1,1]+matrix[1,0]))/2
print(macro)

print(classification_report(target_Val.argmax(axis=1), predictions_bool,digits=3))
