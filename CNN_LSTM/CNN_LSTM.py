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

df = pd.read_pickle("/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdata_preprocessed.pkl")
print('before')

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
df['text'] = df['text'].apply(lambda x: stemmer.stem(x)) # Stem every word.
print('corrected')

df['created_at'] = df['created_at'].astype(str)
df['followers'] = df['followers'].astype(str)
df['following'] = df['following'].astype(str)

cat_Cols = ['text', 'name_user','name_zero_user_mentions_entities', 'location_user', 'description_user', 'contributors_enabled_user', 'default_profile_image_user', 'default_profile_user', 'favorited', 'follow_request_sent_user', 'following_user', 'geo_enabled_user', 'has_extended_profile_user', 'id', 'id_str', 'id_str_user', 'id_str_zero_user_mentions_entities', 'id_user', 'id_zero_user_mentions_entities', 'is_quote_status', 'is_translation_enabled_user', 'is_translator_user', 'lang', 'location_user', 'name_user', 'notifications_user', 'possibly_sensitive', 'possibly_sensitive_appealable', 'profile_background_color_user', 'profile_background_tile_user', 'profile_link_color_user', 'profile_sidebar_border_color_user', 'profile_sidebar_fill_color_user', 'profile_text_color_user', 'profile_use_background_image_user', 'protected_user', 'retweeted', 'screen_name_user', 'screen_name_zero_user_mentions_entities', 'translator_type_user', 'truncated', 'verified_user']


cat_Features =df['text']
print(str(df['text']).encode('utf-8').decode('latin-1'))

for col in cat_Cols:
    df[col] = df[col].astype(str)

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

strings = [] 
stop_words = set(stopwords.words('english')) 

print(df.shape)
for i,line in enumerate(tqdm_notebook(cat_Features, total=df.shape[0])): 
    line = re.sub(r'^https?:\/\/.*[\r\n]*', '', line, flags=re.MULTILINE)
    line = line.replace("'","")
    line = line.replace("\[","")
    line = line.replace(":","")
    line = line.replace("-","")
    line = line.replace("_","")
    line = line.replace("+","")
    line = line.replace("?","")
    line = line.replace(",","")
    line = line.replace("\]","")
    line = line.replace(".","")
    line = line.replace("â€˜","")
    line = line.replace("â€™","")
    line = line.replace("@","")
    line = line.replace("!","")
    line = line.replace("$","")
    line = line.replace("=","")
    line = line.replace("(","")
    line = line.replace(")","")
    line = line.replace("/","")
    line = line.replace("\\","")
    line = line.replace("&","")
    line = line.replace("#","")
    line = re.sub('\d', '', line)
    line = line.split(' ')
    line = [w for w in line if not w in stop_words]
    line = str(line)
    line = str(line.strip())[1:-1].replace(' ', ' ')
    strings.append(line)

#encode text as numbers
tok_Len = 100000 # max number of words for tokenizer
tokenizer = Tokenizer(num_words=tok_Len)
tokenizer.fit_on_texts(strings)
sequences = tokenizer.texts_to_sequences(strings)
term_Index = tokenizer.word_index
print('Number of Terms:', len(term_Index))


sen_Len = 98 # max length of each sentences, including padding
tok_Features = pad_sequences(sequences, padding = 'post', maxlen = sen_Len)
print('Shape of tokenized features tensor:', tok_Features.shape)

indices = np.arange(tok_Features.shape[0])
np.random.shuffle(indices)
time_series = df['created_at_retweets']
time_series.reset_index(drop=True, inplace=True)

time_series = time_series[indices]
tok_Features = tok_Features[indices]
labels = label[indices]

test_Perc = 0.2 #20% data used for testing
num_validation_samples = int(test_Perc*tok_Features.shape[0])
features_Train = tok_Features[: -num_validation_samples]
time_series_Train = time_series[: -num_validation_samples]
num_Norm_Train = num_Norm[: -num_validation_samples]
target_Train = labels[: -num_validation_samples]
features_Val = tok_Features[-num_validation_samples: ]

num_Norm_Val = num_Norm[-num_validation_samples: ]
num_Norm_Val = num_Norm_Val.to_numpy()
print('NUM NORM VAL')
print(num_Norm_Val)
time_series_Val = time_series[-num_validation_samples: ]

print(time_series_Train.shape)
print('TIME SERIES VAL')
print(time_series_Val)
print(time_series_Val.shape)
target_Val = labels[-num_validation_samples: ]
print('Number of records in each attribute:')
print('training: ', target_Train.sum(axis=0))
print('validation: ', target_Val.sum(axis=0))


emb_Dim = 100 # embedding dimensions for word vectors
glove = '/local/home/henrikm/Fakenews_Classification/LSTM_orig/glove.6B.'+str(emb_Dim)+'d.txt'
emb_Ind = {}
f = open(glove, encoding='utf8')
print('Loading Glove \n')
for line in f:
    values = line.split()
    term = values[0]
    emb_Ind[term] = np.asarray(values[1:], dtype='float32')
f.close()
print("Done---\nMap terms to embedding---")

emb_Mat = np.random.random((len(term_Index) + 1, emb_Dim))
for term, i in term_Index.items():
    emb_Vec = emb_Ind.get(term)
    if emb_Vec is not None:
        emb_Mat[i] = emb_Vec
print("Done")

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

features_Train = np.concatenate([features_Train,time_series_Mat],axis=1)

features_Val = np.concatenate([features_Val,time_series_Mat_Val],axis=1)

features_Train = np.asarray(features_Train)
features_Val = np.asarray(features_Val)
for i,features in enumerate(features_Train):
    features_Train[i] = np.array(features_Train[i])
print(features_Train.shape)
print(features_Val.shape)
for i,features in enumerate(features_Val):
    features_Val[i] = np.array(features_Val[i])
print(features_Train.shape)
print(features_Val.shape)
pad_len = max([len(x) for x in features_Train])

for i,features in enumerate(features_Train):

    features_Train[i] = np.pad(features,(0,pad_len-len(features)), 'constant')

features_Pad = np.zeros((len(features_Train),pad_len))

for i, features in enumerate(features_Train):

    for j, feature in enumerate(features_Train[i]):
        features_Pad[i,j] = feature

for i,features in enumerate(features_Val):

    features_Val[i] = np.pad(features,(0,pad_len-len(features)), 'constant')

features_Pad_Val = np.zeros((len(features_Val),pad_len))

for i, features in enumerate(features_Val):

    for j, feature in enumerate(features_Val[i]):
        features_Pad_Val[i,j] = feature

print(features_Pad.shape)

from keras.layers import Input, Bidirectional, Embedding, LSTM, Dense, merge, concatenate, GRU, Conv1D, Flatten, Dropout
from keras.models import Model
from keras import metrics

main_input = Input(shape=(pad_len,), dtype='int32', name='main_input')

x = Embedding(output_dim=512, input_dim=20000, input_length=pad_len)(main_input)

gru_out = Bidirectional(LSTM(32))(x)
gru_out = Dropout(0.5)(gru_out)
auxiliary_output = Dense(2, activation='sigmoid', name='aux_output')(gru_out)

auxiliary_input = Input(shape=(13,), name='aux_input')
e = Embedding(output_dim=512, input_dim=20000, input_length=13)(auxiliary_input)
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
num_Norm_Train = num_Norm_Train.to_numpy()

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
print(time_series_Mat.shape)
model.fit([features_Pad,num_Norm_Train], [target_Train, target_Train], nb_epoch=2, batch_size=32,callbacks=[es])

print(time_series_Mat_Val.shape)
print(num_Norm_Val.shape)
print(type(time_series_Mat_Val))
print(type(num_Norm_Val))
#predictions = model.predict(np.array([[time_series_Mat_Val,num_Norm_Val]]))
predictions = model.predict([features_Pad_Val,num_Norm_Val])
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
import pickle
with open('/local/home/henrikm/Fakenews_Classification/T_Test/CNN_LSTM_accuracies.pkl','wb') as f:
    pickle.dump(predictions[0], f)
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
