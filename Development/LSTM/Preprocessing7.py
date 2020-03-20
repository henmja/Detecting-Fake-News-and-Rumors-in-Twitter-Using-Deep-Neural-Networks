
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
bigdataPKL = pd.read_pickle("bigdata.pkl")
bigdataPKL.tail()

#remove nan columns by threshold
bigdataPKL = bigdataPKL.dropna(axis = 1, thresh=98)
bigdataPKL.head()

import collections
import numpy as np
#bigdataPKL['indices_zero_urls_entities'] = bigdataPKL['indices_zero_urls_entities'].astype(str)
#bigdataPKL['indices_zero_urls_url_entities_user'] = bigdataPKL['indices_zero_urls_url_entities_user'].astype(str)
#bigdataPKL['indices_zero_user_mentions_entities'] = bigdataPKL['indices_zero_user_mentions_entities'].astype(str)
bigdataPKL['followers'] = bigdataPKL['followers'].astype(str)
bigdataPKL['following'] = bigdataPKL['following'].astype(str)
#bigdataPKL['symbols_entities'] = bigdataPKL['symbols_entities'].astype(str)
for col in bigdataPKL.columns:
    if(bigdataPKL[col].dtype == bool):
        bigdataPKL[col].fillna('Unknown', inplace=True)
    elif(bigdataPKL[col].dtype == np.float64):
        bigdataPKL[col].fillna(bigdataPKL[col].mean(), inplace=True)
    elif(bigdataPKL[col].dtype == object):
        #Replace with most common value for rows class
        fakeIndexes = bigdataPKL.index[bigdataPKL['label'] == 'fake'].tolist()
        realIndexes = bigdataPKL.index[bigdataPKL['label'] == 'real'].tolist()
        realVals = []
        fakeVals = []
        for idx in fakeIndexes:
            fakeVals.append(bigdataPKL[col][idx])
        for idx in realIndexes:
            realVals.append(bigdataPKL[col][idx])
        #print(collections.Counter(fakeVals)[0])
        for fakeID in fakeIndexes:
            try:
                if np.isnan(np.float(bigdataPKL[col][fakeID])):
                    while 'nan' in fakeVals: 
                        fakeVals.remove('nan')  
                    while 'NaN' in fakeVals: 
                        fakeVals.remove('NaN') 
                    #print(fakeID)
                    bigdataPKL[col][fakeID] = collections.Counter(fakeVals).most_common()[0][0]
            except ValueError:
                pass
            except TypeError:
                pass
        for realID in realIndexes:
            try:
                if np.isnan(np.float(bigdataPKL[col][realID])):
                    while 'nan' in realVals: 
                        realVals.remove('nan') 
                    while 'NaN' in realVals: 
                        realVals.remove('NaN')  
                    #print(realID)
                    #print(collections.Counter(realVals).most_common()[0][0])
                    bigdataPKL[col][realID] = collections.Counter(realVals).most_common()[0][0]
            except ValueError:
                pass
            except TypeError:
                pass
bigdataPKL.head()

try:
    for row in bigdataPKL['following'].keys(): 
        bigdataPKL['following'][row] = bigdataPKL['following'][row].strip('][')
    for row in bigdataPKL['followers'].keys():
        bigdataPKL['followers'][row] = bigdataPKL['followers'][row].strip('][')
except ValueError:
    pass
bigdataPKL.tail()

bigdataPKL.to_pickle('bigdataClean.pkl')