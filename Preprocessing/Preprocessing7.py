import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
bigdataPKL = pd.read_pickle("bigdata.pkl")
#print(bigdataPKL.tail())
print('check1')
#remove nan columns by threshold
bigdataPKL = bigdataPKL.dropna(axis = 1, thresh=98)
#print(bigdataPKL.head())

import collections
import numpy as np
#bigdataPKL['indices_zero_urls_entities'] = bigdataPKL['indices_zero_urls_entities'].astype(str)
#bigdataPKL['indices_zero_urls_url_entities_user'] = bigdataPKL['indices_zero_urls_url_entities_user'].astype(str)
#bigdataPKL['indices_zero_user_mentions_entities'] = bigdataPKL['indices_zero_user_mentions_entities'].astype(str)
bigdataPKL['followers'] = bigdataPKL['followers'].astype(str)
print('check2')
bigdataPKL['following'] = bigdataPKL['following'].astype(str)

#video_indexes = pd.read_pickle('video_indexes.pkl')
#print(video_indexes)
#df = pd.read_pickle("bigdataClean.pkl")
#df.tail()
#print(bigdataPKL.shape)
#bigdataPKL.drop(bigdataPKL.index[video_indexes],inplace=True)
#print(bigdataPKL.shape)

#norwegian =[]

#for i,rec in enumerate(bigdataPKL['text']):
   # print(rec)
#    try:
#        print(norwegian[-1])
#    except IndexError:
#        print('IndexError')
#    try:
#        rec.encode(encoding='utf-8').decode('ascii')
#    except UnicodeDecodeError:
#        norwegian.append(i)
#bigdataPKL.drop(norwegian, inplace=True)  
print('check3')
bigdataPKL['text'].replace('',np.nan)
bigdataPKL = bigdataPKL[bigdataPKL['text'].notna()]
bigdataPKL = bigdataPKL[~bigdataPKL.text.str.contains('nan',na=False)]
bigdataPKL = bigdataPKL[~bigdataPKL.text.str.contains('NaN',na=False)]
print(bigdataPKL.shape)
#bigdataPKL['symbols_entities'] = bigdataPKL['symbols_entities'].astype(str)
for col in bigdataPKL.columns:
    print(col)
    if(bigdataPKL[col].dtype == bool):
        bigdataPKL[col].fillna('Unknown', inplace=True)
    elif(bigdataPKL[col].dtype == np.float64):
        bigdataPKL[col].fillna(bigdataPKL[col].mean(), inplace=True)
    elif(bigdataPKL[col].dtype == object):
       # bigdataPKL[col] = bigdataPKL[col].transform(lambda x: pd.Series.mode(x)[0])
       bigdataPKL[col].fillna(bigdataPKL[col].mode(), inplace=True)

       
       #Replace with most common value for rows class
        #fakeIndexes = bigdataPKL.index[bigdataPKL['label'] == 'fake'].tolist()
        #realIndexes = bigdataPKL.index[bigdataPKL['label'] == 'real'].tolist()
       # realVals = []
        #fakeVals = []
        #for idx in fakeIndexes:
        #    fakeVals.append(bigdataPKL[col][idx])
        #for idx in realIndexes:
        #    realVals.append(bigdataPKL[col][idx])
        #print(collections.Counter(fakeVals)[0])
        #fakeNANs = []
        #for fakeID in fakeIndexes:
#            print(fakeID)
            #try:
                #if np.isnan(np.float(bigdataPKL[col][fakeID])):
                    #bigdataPKL[col][realID] = collections.Counter(fakeVals).most_common()[0][0]
                    
                    #fakeNANs.append(fakeID)
                    #for item in fakeVals:
                    #    if 'nan' in item:
                    #        fakeVals = fakeVals.remove(item)  
                    #for item in fakeVals:
                    #    if 'NaN' in item:
                    #        fakeVals = fakeVals.remove(item) 
                    #print(fakeID)
                    #bigdataPKL[col][fakeID] = collections.Counter(fakeVals).most_common()[0][0]
            #except ValueError:
            #    pass
            #except TypeError:
            #    pass
        #for fakeID in fakeIndexes:
        #   bigdataPKL[col][fakeID] = collections.Counter(fakeVals).most_common()[0][0] 
        #realNANs = []
        #for realID in realIndexes:
        #    try:
        #        if np.isnan(np.float(bigdataPKL[col][realID])):
#                    realNANs.append(realID) 
        #            bigdataPKL[col][realID] = collections.Counter(realVals).most_common()[0][0]
                    #if collections.Counter(realVals).most_common()[0][0] is 'nan':
                        

                    #if collections.Counter(realVals).most_common()[0][0] is 'NaN':

                    #for item in realVals:
                    #    if 'nan' in item:
                    #        realVals = realVals.remove(item)
                    #for item in realVals:
                    #    if 'NaN' in item:
                    #        realVals = realVals.remove(item)  
                    #print(realID)
                    #print(collections.Counter(realVals).most_common()[0][0])
                    #bigdataPKL[col][realID] = collections.Counter(realVals).most_common()[0][0]
         #   except ValueError:
         #       pass
         #   except TypeError:
         #       pass
#        for realID in realNANs:
#            bigdataPKL[col][realID] = collections.Counter(realVals).most_common()[0][0]
#print(bigdataPKL.head())

try:
    for row in bigdataPKL['following'].keys(): 
        bigdataPKL['following'][row] = bigdataPKL['following'][row].strip('][')
    for row in bigdataPKL['followers'].keys():
        bigdataPKL['followers'][row] = bigdataPKL['followers'][row].strip('][')
except ValueError:
    pass
#print(bigdataPKL.tail())

bigdataPKL.to_pickle('bigdataClean.pkl')

