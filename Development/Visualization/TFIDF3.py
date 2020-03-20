with open('fakeArticlesTermFreq.pickle', 'rb') as handle:
    fakeArticlesTermFreq = pickle.load(handle)
with open('realArticlesTermFreq.pickle', 'rb') as handle:
    realArticlesTermFreq = pickle.load(handle)

with open('fakeIDF.pickle', 'rb') as handle:
    fakeArticlesIDF = pickle.load(handle)
with open('realIDF.pickle', 'rb') as handle:
    realArticlesIDF = pickle.load(handle)
print(fakeArticlesIDF[0])

fakeArticlesTFIDF = {}
realArticlesTFIDF = {}
for art in fakeArticlesTermFreq:
    if art not in fakeArticlesTFIDF.keys():
        fakeArticlesTFIDF[art] = {}
    for word in fakeArticlesTermFreq[art]:
        if word not in fakeArticlesTFIDF[art].keys():
            fakeArticlesTFIDF[art][word] = 0
        try:
            fakeArticlesTFIDF[art][word] = fakeArticlesTermFreq[art][word]*fakeArticlesIDF[art][word]
        except KeyError:
            pass
for art in realArticlesTermFreq:
    if art not in realArticlesTFIDF.keys():
        realArticlesTFIDF[art] = {}
    for word in realArticlesTermFreq[art]:
        if word not in realArticlesTFIDF[art].keys():
            realArticlesTFIDF[art][word] = 0
        try:
            realArticlesTFIDF[art][word] = realArticlesTermFreq[art][word]*realArticlesIDF[art][word]
        except KeyError:
            pass
print(realArticlesTFIDF)

import matplotlib.pyplot as plt
import math
import copy
import numpy as np
import pandas as pd
import seaborn as sns

fakeTFIDF = copy.deepcopy(fakeArticlesTFIDF)
realTFIDF = copy.deepcopy(realArticlesTFIDF)

#MEAN
vec = []
nestVec = []
for art in fakeTFIDF:
    for word in fakeTFIDF[art]:
        nestVec.append(fakeTFIDF[art][word])
    vec.append(nestVec)
    nestVec=[]
vec = [item for sublist in vec for item in sublist]
fakeMean = np.mean(vec)

vec = []
nestVec = []
for art in realTFIDF:
    for word in realTFIDF[art]:
        nestVec.append(realTFIDF[art][word])
    vec.append(nestVec)
    nestVec=[]
vec = [item for sublist in vec for item in sublist]
realMean = np.mean(vec)

for key in fakeTFIDF:
    if key not in realTFIDF.keys():
        realTFIDF[key]={}
        realTFIDF[key]['null']=np.nan
for key in realTFIDF:
    if key not in fakeTFIDF.keys():
        fakeTFIDF[key]={}
        fakeTFIDF[key]['null']=np.nan
        
fig = plt.figure()
ax = fig.add_subplot(111)

y1 = []
y_temp=[]

    
    
for art in realTFIDF:
    for word in realTFIDF[art]:
        y_temp.append(realTFIDF[art][word])
    y1.append(y_temp)
    y_temp=[]
y1 = [item for sublist in y1 for item in sublist]


y2 = []

y_temp=[]
for art in fakeTFIDF:
    for word in fakeTFIDF[art]:
        y_temp.append(fakeTFIDF[art][word])
    y2.append(y_temp)
    y_temp=[]
y2 = [item for sublist in y2 for item in sublist]

print(fakeMean)
print(realMean)

#CDF
x = np.sort(y1)
n = x.size
y = np.arange(1, n+1) / n

ax.scatter(x=x, y=y);

x = np.sort(y2)
n = x.size
y = np.arange(1, n+1) / n

ax.scatter(x=x, y=y);

ax.legend(('Real','Fake'))
ax.set_xlabel('TF-IDF')
ax.set_ylabel('Probability')
fig.suptitle('TF-IDF CDF', fontsize=20)
plt.xlim(0,0.8)
plt.show()
plt.savefig("TF-IDF CDF"+".png", bbox_inches='tight')

!pip install wordcloud

#TF-IDF Wordcloud
#ONLY ADD ENGLISH WORDS TO TOP 50
from wordcloud import WordCloud
#top 50 TF-IDF Array
import copy
real_TFIDF = copy.deepcopy(realArticlesTFIDF)
fake_TFIDF = copy.deepcopy(fakeArticlesTFIDF)
#print(real_TFIDF[1])
top_50_real_tfidf = []
top_50_real_tfidf_val = []
temp = ''
for i in range(50):
    print(i)
    top_50_real_tfidf.append('')
    top_50_real_tfidf_val.append(0)
    for user in real_TFIDF:
        #print(user)
        try:
            for tfidf in real_TFIDF[user]:
            #print(real_TFIDF[user])
            #if tfidf==list(real_TFIDF[user].keys())[-1]:
            #    print(tfidf)
            #print(tfidf)
                if real_TFIDF[user][tfidf]>=top_50_real_tfidf_val[i]:
                    top_50_real_tfidf[i] = tfidf
                    #temp = user
        except TypeError:
            pass
    for user in real_TFIDF:
        try:
            real_TFIDF[user]=real_TFIDF[user].pop(top_50_real_tfidf[i])
        except KeyError:
            pass
        except AttributeError:
            pass
    #try:
    #    if user==list(real_TFIDF.keys())[-1] and tfidf==list(real_TFIDF[user].keys())[-1]:
    #        print(top_50_real_tfidf[i])
    #        try:
            #print(top_50_real_tfidf[i])
    #            real_TFIDF[temp]=real_TFIDF[temp].pop(top_50_real_tfidf[i])
    #        except IndexError:
    #            pass
    #except AttributeError:
    #    pass
top_50_fake_tfidf = []
top_50_fake_tfidf_val = []
temp = ''
for i in range(50):
    top_50_fake_tfidf.append('')
    top_50_fake_tfidf_val.append(0)
    for user in fake_TFIDF:
        try:
            for tfidf in fake_TFIDF[user]:
            #print(fake_TFIDF[user])
            #print(user)
            #print(tfidf)
            #print(i)
                if fake_TFIDF[user][tfidf]>=top_50_fake_tfidf_val[i]:
                    top_50_fake_tfidf[i] = tfidf
                    #temp = user
        except TypeError:
            pass
    for user in fake_TFIDF:
        try:
            fake_TFIDF[user].pop(top_50_fake_tfidf[i])
        except KeyError:
            pass
        except AttributeError:
            pass
    #try:
    #    if user==list(fake_TFIDF.keys())[-1] and tfidf==list(real_TFIDF[user].keys())[-1]:
    #        try:
    #            fake_TFIDF[temp].pop(top_50_fake_tfidf[i])
    #        except IndexError:
    #            pass
    #except AttributeError:
    #    pass
#print(top_50_real_tfidf)
import matplotlib.pyplot as plt

#convert list to string and generate
unique_string=(" ").join(top_50_real_tfidf)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("real_TFIDF_Wordcloud"+".png", bbox_inches='tight')


#convert list to string and generate
unique_string=(" ").join(top_50_fake_tfidf)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("fake_TFIDF_Wordcloud"+".png", bbox_inches='tight')


import numpy as np
#TF-IDF Wordcloud
#ONLY ADD ENGLISH WORDS TO TOP 50
from wordcloud import WordCloud
#top 50 TF-IDF Array
import copy
real_TFIDF = copy.deepcopy(realArticlesTFIDF)
fake_TFIDF = copy.deepcopy(fakeArticlesTFIDF)
#print(real_TFIDF[1])
top_50_real_tfidf = []
top_50_real_tfidf_val = []
temp = ''
for i in range(50):
    max_Appended = False
    print(i)
    top_50_real_tfidf.append('')
    for user in real_TFIDF:
        #print(user)
        try:
            max_tfidf = max(real_TFIDF[user].values())
            max_keys = [k for k, v in real_TFIDF[user].items() if v == max_tfidf]
            top_50_real_tfidf[i] = max_keys[0]
            if max_Appended==False:
                top_50_real_tfidf_val.append(max_keys[0])
                max_Appended=True
            try:
                for tfidf in real_TFIDF[user]:
            #print(real_TFIDF[user])
            #if tfidf==list(real_TFIDF[user].keys())[-1]:
            #    print(tfidf)
            #print(tfidf)
                    if real_TFIDF[user][tfidf]<=top_50_real_tfidf_val[i]:
                        top_50_real_tfidf[i] = tfidf    
            except TypeError:
                pass
        except AttributeError:
            pass
    for user in real_TFIDF:
        try:
            real_TFIDF[user]=real_TFIDF[user].pop(top_50_real_tfidf[i])
        except KeyError:
            pass
        except AttributeError:
            pass
    #try:
    #    if user==list(real_TFIDF.keys())[-1] and tfidf==list(real_TFIDF[user].keys())[-1]:
    #        print(top_50_real_tfidf[i])
    #        try:
            #print(top_50_real_tfidf[i])
    #            real_TFIDF[temp]=real_TFIDF[temp].pop(top_50_real_tfidf[i])
    #        except IndexError:
    #            pass
    #except AttributeError:
    #    pass
top_50_fake_tfidf = []
top_50_fake_tfidf_val = []
temp = ''
for i in range(50):
    max_Appended = False
    top_50_fake_tfidf.append('')
    for user in fake_TFIDF:
        try:
            max_tfidf = max(fake_TFIDF[user].values())
            max_keys = [k for k, v in fake_TFIDF[user].items() if v == max_tfidf]
            top_50_fake_tfidf[i] = max_keys[0]
            if max_Appended==False:
                top_50_fake_tfidf_val.append(max_keys[0])
                max_Appended=True
            try:
                for tfidf in fake_TFIDF[user]:
            #print(fake_TFIDF[user])
            #print(user)
            #print(tfidf)
            #print(i)
                    max_tfidf = max(fake_TFIDF[user].values())
                    max_keys = [k for k, v in fake_TFIDF[user].items() if v == max_tfidf]
                    top_50_fake_tfidf[i] = max_keys[0]
                    if fake_TFIDF[user][tfidf]<=top_50_fake_tfidf_val[i]:
                        top_50_fake_tfidf[i] = tfidf     
            except TypeError:
                pass
        except AttributeError:
            pass
    for user in fake_TFIDF:
        try:
            fake_TFIDF[user]=fake_TFIDF[user].pop(top_50_fake_tfidf[i])
        except KeyError:
            pass
        except AttributeError:
            pass
    #try:
    #    if user==list(fake_TFIDF.keys())[-1] and tfidf==list(real_TFIDF[user].keys())[-1]:
    #        try:
    #            fake_TFIDF[temp].pop(top_50_fake_tfidf[i])
    #        except IndexError:
    #            pass
    #except AttributeError:
    #    pass
#print(top_50_real_tfidf)
import matplotlib.pyplot as plt

#convert list to string and generate
unique_string=(" ").join(top_50_real_tfidf)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("real_TFIDF_Common_Wordcloud"+".png", bbox_inches='tight')


#convert list to string and generate
unique_string=(" ").join(top_50_fake_tfidf)
wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)
plt.figure(figsize=(15,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("fake_TFIDF_Common_Wordcloud"+".png", bbox_inches='tight')