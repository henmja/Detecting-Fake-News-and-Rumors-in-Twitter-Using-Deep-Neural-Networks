#coding: utf-8
import json
import matplotlib.pyplot as plt
from keras.layers import Bidirectional
from langdetect import detect
from  tqdm import tqdm
import nltk
nltk.download('stopwords')
from sklearn.metrics import roc_auc_score
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
from keras.layers import Bidirectional
from keras.layers import Conv1D
from imblearn.under_sampling import RandomUnderSampler
from keras.layers import GlobalMaxPool1D
from keras.layers import Dropout
from keras.layers import InputLayer
from sklearn.metrics import classification_report
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
df = pd.read_pickle("../Preprocessing/bigdata_preprocessed.pkl")
df['created_at'] = df['created_at'].astype(str)
df['followers'] = df['followers'].astype(str)
df['following'] = df['following'].astype(str)

cat_Cols = ['text', 'name_user','name_zero_user_mentions_entities', 'location_user', 'description_user', 'contributors_enabled_user', 'default_profile_image_user', 'default_profile_user', 'favorited', 'follow_request_sent_user', 'following_user', 'geo_enabled_user', 'has_extended_profile_user', 'id', 'id_str', 'id_str_user', 'id_str_zero_user_mentions_entities', 'id_user', 'id_zero_user_mentions_entities', 'is_quote_status', 'is_translation_enabled_user', 'is_translator_user', 'lang', 'location_user', 'name_user', 'notifications_user', 'possibly_sensitive', 'possibly_sensitive_appealable', 'profile_background_color_user', 'profile_background_tile_user', 'profile_link_color_user', 'profile_sidebar_border_color_user', 'profile_sidebar_fill_color_user', 'profile_text_color_user', 'profile_use_background_image_user', 'protected_user', 'retweeted', 'screen_name_user', 'screen_name_zero_user_mentions_entities', 'translator_type_user', 'truncated', 'verified_user']

cat_Features =df['text']
print(str(df['text']).encode('utf-8').decode('latin-1'))
for col in cat_Cols:
    df[col] = df[col].astype(str)

print(df['label'])
label = pd.get_dummies(df['label'])
label = np.array(label)
print(np.argmax(label, axis=1))

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


sen_Len = 162 # max length of each sentences, including padding
tok_Features = pad_sequences(sequences, padding = 'post', maxlen = sen_Len-111)
print('Shape of tokenized features tensor:', tok_Features.shape)

indices = np.arange(tok_Features.shape[0])
np.random.shuffle(indices)
time_series = df['created_at_retweets']
time_series.reset_index(drop=True, inplace=True)
print('before')
for i,time in enumerate(time_series):
    if isinstance(time,float):
        print(i)
        print(time)
print('etter')
print(type(time_series))
time_series = time_series[indices]
print('before2')
print(time_series)
for i,time in enumerate(time_series):
    #if isinstance(time,float):
    print(i)
    print(type(time))
    print('etter2')
tok_Features = tok_Features[indices]
labels = label[indices]
print(num_Norm)
print(num_Norm.shape)
print(type(num_Norm))


test_Perc = 0.2 #20% data used for testing
num_validation_samples = int(test_Perc*tok_Features.shape[0])
features_Train = tok_Features[: -num_validation_samples]
time_series_Train = time_series[: -num_validation_samples]
num_Norm_Train = num_Norm[: -num_validation_samples]
target_Train = labels[: -num_validation_samples]
features_Val = tok_Features[-num_validation_samples: ]
num_Norm_Val = num_Norm[-num_validation_samples: ]
time_series_Val = time_series[-num_validation_samples: ]
target_Val = labels[-num_validation_samples: ]
print('Number of records in each attribute:')
print('training: ', target_Train.sum(axis=0))
print('validation: ', target_Val.sum(axis=0))


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
#    print('type emb_Vec')
#    print(type(emb_Vec))
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

features_Train = np.concatenate([features_Train,num_Norm_Train],axis=1)

undersample = RandomUnderSampler()

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
print(target_Train)
print(target_Train.shape)
print(type(target_Train))
features_Train_Resampled, target_Train_Resampled = undersample.fit_resample(features_Train, target_Train)
from keras.utils import to_categorical
target_Train_Resampled = to_categorical(target_Train_Resampled)
print(target_Train.shape)
print(type(target_Train))
print(target_Train)
print('FEATURES TRAIN')
def create_model():
	# create model
    model = Sequential()
    model.add(InputLayer((sen_Len,),dtype='int32'))
    e = Embedding(len(term_Index) + 1, emb_Dim, weights=[emb_Mat], input_length=sen_Len, trainable=False)
    print('SEN LEN')
    print(sen_Len)
    print('TERM INDEX')
    print(len(term_Index))
    model.add(e)
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.1))
    model.add(Dense(158, activation='relu'))
    model.add(Dropout(0.1))
#prevents blowing up activation
    model.add(Dense(2, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['acc'])
    return model

gridmodel = KerasClassifier(build_fn=create_model,epochs=10, batch_size=5, verbose=0)

batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=gridmodel, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(features_Train_Resampled, target_Train_Resampled)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print(type(grid_result.best_params_))
print(len(grid_result.best_params_))
print(grid_result.best_params_.keys())
epochs_var = grid_result.best_params_['epochs']
batch_size_var = grid_result.best_params_['batch_size']

model = Sequential()
model.add(InputLayer((sen_Len,),dtype='int32'))
e = Embedding(len(term_Index) + 1, emb_Dim, weights=[emb_Mat], input_length=sen_Len, trainable=False)
print('SEN LEN')
print(sen_Len)
print('TERM INDEX')
print(len(term_Index))
model.add(e)
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(158, activation='relu'))
model.add(Dropout(0.1))
#prevents blowing up activation
model.add(Dense(2, activation='sigmoid'))
    
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['acc'])

print(max_len)
history = model.fit(features_Train_Resampled, target_Train_Resampled, epochs = epochs_var, batch_size=batch_size_var, validation_split=0.20)
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
combined_Val = np.concatenate([combined_Val,num_Norm_Val],axis=1)

target_Val = np.argmax(target_Val, axis=1)

predictions = model.predict(combined_Val)
predictions_bool = np.argmax(predictions, axis=1)
print(predictions_bool.shape)
print(classification_report(target_Val, predictions_bool,digits=3))
predictions_prob = model.predict_proba(combined_Val)
target_Val = to_categorical(target_Val)
auc = roc_auc_score(target_Val, predictions_prob)
import sklearn.metrics as metrics
y_pred = (predictions > 0.5)
matrix = metrics.confusion_matrix(target_Val.argmax(axis=1), y_pred.argmax(axis=1))
micro = (matrix[0,0]+matrix[1,1])/(matrix[0,0]+matrix[1,1]+matrix[0,1]+matrix[1,0])
print('micro')
print(micro)
print('macro')
macro = (matrix[0,0]/(matrix[0,0]+matrix[1,0])+matrix[1,1]/(matrix[1,1]+matrix[1,0]))/2
print(macro)
print('auc')
print(auc)
import sklearn.metrics as metrics
y_pred = (predictions > 0.5)
matrix = metrics.confusion_matrix(target_Val.argmax(axis=1), y_pred.argmax(axis=1))
print(matrix)
import matplotlib.pyplot as plt
#%matplotlib inline
val_Loss = history.history['val_loss']
loss = history.history['loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_Loss, label='Testing Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("loss_Function"+".png", bbox_inches='tight')
acc = history.history['acc']
val_Acc = history.history['val_acc']
plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_Acc, label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig("accuracy"+".png", bbox_inches='tight')
print(matrix)
