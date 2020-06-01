#coding: utf-8
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import json
import matplotlib.pyplot as plt
from langdetect import detect  #detects what language is written
from  tqdm import tqdm
import nltk
nltk.download('stopwords')
from sklearn import svm
import nltk
import re
from tqdm import tqdm_notebook
from nltk.corpus import stopwords

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import GlobalMaxPool1D
from keras.layers import Dropout
from keras.layers import InputLayer
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
df = pd.read_pickle("../Preprocessing/bigdata_preprocessed.pkl")
df['created_at'] = df['created_at'].astype(str)
df['followers'] = df['followers'].astype(str)
df['following'] = df['following'].astype(str)

cat_Cols = ['text', 'name_user','name_zero_user_mentions_entities', 'location_user', 'description_user', 'contributors_enabled_user', 'default_profile_image_user', 'default_profile_user', 'favorited', 'follow_request_sent_user', 'following_user', 'geo_enabled_user', 'has_extended_profile_user', 'id', 'id_str', 'id_str_user', 'id_str_zero_user_mentions_entities', 'id_user', 'id_zero_user_mentions_entities', 'is_quote_status', 'is_translation_enabled_user', 'is_translator_user', 'lang', 'location_user', 'name_user', 'notifications_user', 'possibly_sensitive', 'possibly_sensitive_appealable', 'profile_background_color_user', 'profile_background_tile_user', 'profile_link_color_user', 'profile_sidebar_border_color_user', 'profile_sidebar_fill_color_user', 'profile_text_color_user', 'profile_use_background_image_user', 'protected_user', 'retweeted', 'screen_name_user', 'screen_name_zero_user_mentions_entities', 'translator_type_user', 'truncated', 'verified_user']


cat_Features =df['text']
print(str(df['text']).encode('utf-8').decode('latin-1'))
for col in cat_Cols:
    df[col] = df[col].astype(str)


label = df['label']
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
import pandas as pd
from sklearn import preprocessing

x = mat.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
x_scaled = min_max_scaler.fit_transform(x)
num_Norm = pd.DataFrame(x_scaled)

for col in num_Norm.keys():
    num_Norm[col] = pd.cut(num_Norm[col], bins=3, labels=np.arange(3), right=False)


strings = [] 
stop_words = set(stopwords.words('english')) 
    
for line in tqdm_notebook(cat_Features, total=df.shape[0]):
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


sen_Len = 150 # max length of each sentences, including padding
tok_Features = pad_sequences(sequences, padding = 'post', maxlen = sen_Len-98)
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
num_norm_Train = num_Norm[: -num_validation_samples]
target_Train = labels[: -num_validation_samples]
features_Val = tok_Features[-num_validation_samples: ]
time_series_Val = time_series[-num_validation_samples: ]
num_norm_Val = num_Norm[-num_validation_samples: ]
target_Val = labels[-num_validation_samples: ]
print('Number of records in each attribute:')



emb_Dim = 100 # embedding dimensions for word vectors
glove = '../LSTM/glove.6B.'+str(emb_Dim)+'d.txt'
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
print('SHAPE EMB MAT')
print(emb_Mat.shape)
print(type(emb_Mat))
print('SHAPE TIME SERIES')
print(time_series_Train.shape)
print(type(time_series_Train))
maxLen = 0
maxIndex = 0

time_series_Train = np.asarray(time_series_Train)

for i,times in enumerate(time_series_Train):

    time_series_Train[i] = np.array(time_series_Train[i])
max_len = max([len(x) for x in time_series_Train]) 
for i,times in enumerate(time_series_Train):
    time_series_Train[i] = np.pad(times,(0,max_len-len(times)), 'constant')
time_series_Mat = np.zeros((len(time_series_Train),max_len))
for i, times in enumerate(time_series_Train):
    for j, time in enumerate(time_series_Train[i]):
        time_series_Mat[i,j] = time
features_Train = np.concatenate([features_Train,time_series_Mat],axis=1)
features_Train = np.concatenate([features_Train,num_norm_Train],axis=1)

target_Train = target_Train.reset_index(drop=True)
target_Train = target_Train.dropna()
print(features_Train.shape)
print(target_Train.shape)
print(target_Train)
features_Train = features_Train[target_Train.index.values]
oversample = SMOTE()
features_Train_Resampled, target_Train_Resampled = oversample.fit_resample(features_Train, target_Train)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', verbose=2, max_iter=10000,probability=True)
history = SVM.fit(features_Train_Resampled,target_Train_Resampled)
time_series_Val = np.asarray(time_series_Val)

for i,times in enumerate(time_series_Val):
    time_series_Val[i] = np.array(time_series_Val[i])
max_len_val = max([len(x) for x in time_series_Val])
for i,times in enumerate(time_series_Val):
    time_series_Val[i] = np.pad(times,(0,max_len_val-len(times)), 'constant')
time_series_val_Mat = np.zeros((len(time_series_Val),max(max_len,max_len_val)))
for i, times in enumerate(time_series_Val):
    for j, time in enumerate(time_series_Val[i]):
        time_series_val_Mat[i,j] = time
print(time_series_val_Mat)
print(time_series_val_Mat.shape)
combined_Val = np.concatenate([features_Val,time_series_val_Mat],axis=1)
combined_Val = np.concatenate([combined_Val,num_norm_Val],axis=1)
target_Val = target_Val.reset_index(drop=True)
target_Val = target_Val.dropna()
combined_Val = combined_Val[target_Val.index.values]

combined_Val_Resampled, target_Val_Resampled = oversample.fit_resample(combined_Val, target_Val)


predictions_SVM = SVM.predict(combined_Val_Resampled)
print(predictions_SVM)
print(accuracy_score(predictions_SVM, target_Val_Resampled)*100)

import sklearn.metrics as metrics
print('F1 score')
print(classification_report(target_Val_Resampled, predictions_SVM,digits=3))

predictions_prob = SVM.predict_proba(combined_Val_Resampled)
predictions_prob = np.argmax(predictions_prob, axis=1)
auc = roc_auc_score(target_Val_Resampled, predictions_prob)

matrix = metrics.confusion_matrix(target_Val_Resampled, predictions_SVM)

print(matrix)
micro = (matrix[0,0]+matrix[1,1])/(matrix[0,0]+matrix[1,1]+matrix[0,1]+matrix[1,0])
print('micro')
print(micro)
print('macro')
macro = (matrix[0,0]/(matrix[0,0]+matrix[1,0])+matrix[1,1]/(matrix[1,1]+matrix[1,0]))/2
print(macro)
print('auc')
print(auc)

import matplotlib.pyplot as plt
#%matplotlib inline
