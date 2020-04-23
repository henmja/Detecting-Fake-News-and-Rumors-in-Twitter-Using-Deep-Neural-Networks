#coding: utf-8
import pickle
import pandas as pd
#with open('../Visualization/fakeTFIDF.pickle', 'rb') as handle:
#fakeTFIDF = pickle.load(handle)

#with open('../Visualization/realTFIDF.pickle', 'rb') as handle:
#    realTFIDF = pickle.load(handle)
#df_real_tfidf = pd.DataFrame(realTFIDF) 
#df_fake_tfidf = pd.DataFrame(fakeTFIDF)
#df_real_tfidf = df_real_tfidf.transpose()
#df_fake_tfidf = df_fake_tfidf.transpose()
#


#df_tfidf = df_real_tfidf.append(df_fake_tfidf, ignore_index=True)
#print(df_tfidf.shape)
records_per_article = pd.read_pickle('records_per_article.pkl')
#df_tfidf_expanded = pd.DataFrame()
##print(len(records_per_article))
#for i,records in enumerate(records_per_article):
#    print(i)
#    for rec in range(records):
#        df_tfidf_expanded = df_tfidf_expanded.append(df_tfidf.iloc[[i]], ignore_index=True)
#print(df_tfidf_expanded.shape)

df = pd.read_pickle('bigdataClean.pkl')

#df['tfidf'] = df_tfidf_expanded.values.tolist()
#df['tfidf'] = df['tfidf'].apply(lambda x: np.array(x))

video_indexes = pd.read_pickle('video_indexes.pkl')
#print(video_indexes)
#df = pd.read_pickle("bigdataClean.pkl")
#df.tail()
print(df.shape)
df.drop(df.index[video_indexes],inplace=True)
print(df.shape)
df =df[~df.text.str.contains(u"æ",na=False)]
df = df[~df.text.str.contains(u"ø", na=False)]
df = df[~df.text.str.contains(u"å",na=False)]

df.to_pickle('bigdataClean_2.pkl')
