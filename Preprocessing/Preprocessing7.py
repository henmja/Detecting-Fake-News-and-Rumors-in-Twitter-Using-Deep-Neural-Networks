import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
bigdataPKL = pd.read_pickle("/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdata.pkl")
print('check1')
#remove nan columns by threshold
bigdataPKL = bigdataPKL.dropna(axis = 1, thresh=98)

import collections
import numpy as np
bigdataPKL['followers'] = bigdataPKL['followers'].astype(str)
bigdataPKL['following'] = bigdataPKL['following'].astype(str)
 
bigdataPKL['text'].replace('',np.nan)
bigdataPKL = bigdataPKL[bigdataPKL['text'].notna()]
bigdataPKL = bigdataPKL[~bigdataPKL.text.str.contains('nan',na=False)]
bigdataPKL = bigdataPKL[~bigdataPKL.text.str.contains('NaN',na=False)]
print(bigdataPKL.shape)
for col in bigdataPKL.columns:
    if(bigdataPKL[col].dtype == bool):
        bigdataPKL[col].fillna('Unknown', inplace=True)
    elif(bigdataPKL[col].dtype == np.float64):
        bigdataPKL[col].fillna(bigdataPKL[col].mean(), inplace=True)
    elif(bigdataPKL[col].dtype == object):
       bigdataPKL[col].fillna(bigdataPKL[col].mode(), inplace=True)

try:
    for row in bigdataPKL['following'].keys(): 
        bigdataPKL['following'][row] = bigdataPKL['following'][row].strip('][')
    for row in bigdataPKL['followers'].keys():
        bigdataPKL['followers'][row] = bigdataPKL['followers'][row].strip('][')
except IndexError:
    pass
except ValueError:
    pass

bigdataPKL.to_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdataClean.pkl')

