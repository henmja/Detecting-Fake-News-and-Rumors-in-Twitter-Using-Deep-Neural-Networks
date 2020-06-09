#coding: utf-8
import pickle
import pandas as pd

records_per_article = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/records_per_article.pkl')


df = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdataClean.pkl')


video_indexes = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/video_indexes.pkl')
print(df.shape)
try:
    df.drop(df.index[video_indexes],inplace=True)
except IndexError:
    pass
print(df.shape)
df =df[~df.text.str.contains(u"æ",na=False)]
df = df[~df.text.str.contains(u"ø", na=False)]
df = df[~df.text.str.contains(u"å",na=False)]

df.to_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdataClean_2.pkl')
