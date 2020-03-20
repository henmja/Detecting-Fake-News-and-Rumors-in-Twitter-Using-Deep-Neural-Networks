!pip install inflect


import nltk
nltk.download('stopwords')

import nltk
import re
from tqdm import tqdm_notebook
from nltk.corpus import stopwords
from tensorflow.keras import regularizers, initializers, optimizers, callbacks
#add padding so that output has same length as the original input. 
from tensorflow.keras.preprocessing.sequence import pad_sequences
#tokenize words (map strings to integers).
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import *
from keras.models import Sequential
#dense layers are the hidden layers
from keras.layers import Dense
#Convert tensor shape to 1D
from keras.layers import Flatten
#Embedding layer creates word vectors (each word is represented by an integer)
from keras.layers import Embedding
from keras.layers import LSTM
#max pooling discretizes hidden layers output in order to downsample it. This is done to prevent overfitting and decrease
#computational cost.
#Global max pooling = ordinary max pooling layer with pool size equals to the size of the input (minus filter size + 1, to 
#be precise). normal max pooling takes argument for pool size. Global is common for NLP, while normal is for computer vision.
from keras.layers import GlobalMaxPool1D
#Dropout layer is used to prevent overfitting by setting a fraction of input neurons to 0 at each update during training.
from keras.layers import Dropout
from keras.layers import InputLayer
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import numpy as np
df = pd.read_pickle("bigdataClean.pkl")
df.tail()

df['created_at'] = df['created_at'].astype(str)
df['followers'] = df['followers'].astype(str)
df['following'] = df['following'].astype(str)

cat_Cols = ['text', 'name_user','name_zero_user_mentions_entities', 'location_user', 'description_user', 'contributors_enabled_user', 'default_profile_image_user', 'default_profile_user', 'favorited', 'follow_request_sent_user', 'following_user', 'geo_enabled_user', 'has_extended_profile_user', 'id', 'id_str', 'id_str_user', 'id_str_zero_user_mentions_entities', 'id_user', 'id_zero_user_mentions_entities', 'is_quote_status', 'is_translation_enabled_user', 'is_translator_user', 'lang', 'location_user', 'name_user', 'notifications_user', 'possibly_sensitive', 'possibly_sensitive_appealable', 'profile_background_color_user', 'profile_background_tile_user', 'profile_link_color_user', 'profile_sidebar_border_color_user', 'profile_sidebar_fill_color_user', 'profile_text_color_user', 'profile_use_background_image_user', 'protected_user', 'retweeted', 'screen_name_user', 'screen_name_zero_user_mentions_entities', 'translator_type_user', 'truncated', 'verified_user']


cat_Features = ' text ' + df['text'] + ' username ' + df['name_user']
i = 0
for col in cat_Cols:
    df[col] = df[col].astype(str)
for idx in cat_Cols:
    if i>1:
        cat_Features = cat_Features + ' - ' + str(idx) + ': ' + df[idx]
    i=i+1
#x = df[xCols].apply(lambda row: .join(row.values.astype(str)), axis=1)
print(cat_Features[0])


label = pd.get_dummies(df['label'])
label = np.array(label)

strings = [] 
stop_words = set(stopwords.words('english')) 
print(cat_Features[0])
    
#line = re.sub('\d', '', line)
for line in tqdm_notebook(cat_Features, total=df.shape[0]): 
    #PRØV CLEAN TEXT HVIS IKKE FUNGERER
    #strings.append(remove_Symbols(line))
    #print(line)
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
    line = line.replace("‘","")
    line = line.replace("’","")
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
    #print(line)
    #line = [x for x in line if x not in stopwords.english]
    #print(line)
    line = str(line)
    line = str(line.strip())[1:-1].replace(' ', ' ')
    strings.append(line)

for i, row in enumerate(strings):
    strings[i] = row+ ' created at user ' + df['created_at_user'][i]
for i, row in enumerate(strings):
    strings[i] = row + ' created at ' + df['created_at'][i]

#encode text as numbers
tok_Len = 100000 # max number of words for tokenizer
tokenizer = Tokenizer(num_words=tok_Len)
tokenizer.fit_on_texts(strings)
sequences = tokenizer.texts_to_sequences(strings)
term_Index = tokenizer.word_index
print('Number of Terms:', len(term_Index))


sen_Len = 1000 # max length of each sentences, including padding
tok_Features = pad_sequences(sequences, padding = 'post', maxlen = sen_Len)
print('Shape of tokenized features tensor:', tok_Features.shape)
#print('Shape of label tensor:', label.shape)

indices = np.arange(tok_Features.shape[0])
np.random.shuffle(indices)
tok_Features = tok_Features[indices]
labels = label[indices]



test_Perc = 0.2 #20% data used for testing
num_validation_samples = int(test_Perc*tok_Features.shape[0])
features_Train = tok_Features[: -num_validation_samples]
target_Train = labels[: -num_validation_samples]
features_Val = tok_Features[-num_validation_samples: ]
target_Val = labels[-num_validation_samples: ]
print('Number of records in each attribute:')
print('training: ', target_Train.sum(axis=0))
print('validation: ', target_Val.sum(axis=0))


emb_Dim = 100 # embedding dimensions for word vectors
glove = 'glove.6B.'+str(emb_Dim)+'d.txt'
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


model = Sequential()
model.add(InputLayer((sen_Len,),dtype='int32'))
e = Embedding(len(term_Index) + 1, emb_Dim, weights=[emb_Mat], input_length=sen_Len, trainable=False)
model.add(e)
model.add(LSTM(60, return_sequences=True,name='lstm_layer'))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
#Must use relu first to avoid vanishing gradient (https://towardsdatascience.com/is-relu-after-sigmoid-bad-661fda45f7a2)
#relu prevents vanishing gradient (https://stats.stackexchange.com/questions/126238/what-are-the-advantages-of-relu-over-sigmoid-function-in-deep-neural-networks)
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.1))
#prevents blowing up activation
model.add(Dense(2, activation='sigmoid'))




model.compile(optimizer='adam', loss='binary_crossentropy',
 metrics = ['accuracy'])



history = model.fit(features_Train, target_Train, epochs = 50, batch_size=128, validation_split=0.20)
with open('/trainHistoryDict', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

predictions = model.predict(features_Val)


import sklearn.metrics as metrics
y_pred = (predictions > 0.5)
matrix = metrics.confusion_matrix(target_Val.argmax(axis=1), y_pred.argmax(axis=1))
fig = plt.figure()
plt.matshow(matrix)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicated Label')
plt.savefig(matrix+'.jpg')

import matplotlib.pyplot as plt
%matplotlib inline

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

###LAGRE matrix (confusion mattrix)