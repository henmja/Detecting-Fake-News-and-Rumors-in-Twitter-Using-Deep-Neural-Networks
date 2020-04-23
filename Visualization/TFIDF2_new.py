#coding: utf-8
import pickle
import pandas as pd

fakeArticles = pd.read_pickle("fakeArticlesPD4.pkl")
realArticles = pd.read_pickle("realArticlesPD4.pkl")
fakeArticles['label'] = 'fake'
realArticles['label'] = 'real'
video_indexes_tfidf = pd.read_pickle('../LSTM/video_indexes_tfidf.pkl')
#print(video_indexes)
#df = pd.read_pickle("bigdataClean.pkl")
#df.tail()i
#fake_video_indexes = video_indexes[:]
bigdata = realArticles.append(fakeArticles, ignore_index=True)
print(bigdata.shape)
bigdata.drop(bigdata.index[video_indexes_tfidf],inplace=True)
fakeArticles = bigdata[~bigdata.label.str.contains('real')]
realArticles = bigdata[~bigdata.label.str.contains('fake')]
import re
for row in fakeArticles['text'].keys():
    fakeArticles['text'][row]=fakeArticles['text'][row].replace('335 SHARES SHARE THIS STORY','')
    fakeArticles['text'][row]=fakeArticles['text'][row].replace('Vi bruker informasjonskapsler for Ã¥ gjÃ¸re innholdet mer personlig, skreddersy og mÃ¥le annonser og tilby en tryggere opplevelse. NÃ¥r du klikker eller navigerer pÃ¥ nettstedet, godtar du at vi henter inn informasjon pÃ¥ og utenfor Facebook gjennom informasjonskapsler. Finn ut mer, blant annet om tilgjengelige kontroller, her: retningslinjer for informasjonskapsler','')
    fakeArticles['text'][row]=fakeArticles['text'][row].replace('Loading','')
    fakeArticles['text'][row]=fakeArticles['text'][row].replace('Please enable cookies on your web browser in order to continue','')
    fakeArticles['text'][row] = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?Â«Â»â€œâ€â€˜â€™]))''', " ", fakeArticles['text'][row])


#fakeArticles['text'][row] = re.sub('http://\S+|https://\S+', '', fakeArticles['text'][row])
#fakeArticles['text'][row] = re.sub('http[s]?://\S+', '',fakeArticles['text'][row])
#fakeArticles = fakeArticles[~fakeArticles.text.str.contains("http")]
#fakeArticles = fakeArticles[~fakeArticles.text.str.contains(".com")]

print(fakeArticles['text'])

import re
for row in realArticles['text'].keys():
    realArticles['text'][row]=realArticles['text'][row].replace('Automatisk avspilling NÃ¥r automatisk avspilling er pÃ¥, blir en foreslÃ¥tt video automatisk spilt av som neste video','')
    realArticles['text'][row]=realArticles['text'][row].replace('Neste','')
    realArticles['text'][row]=realArticles['text'][row].replace('COPYRIGHT Â© 2005 LexisNexis, a division of Reed Elsevier Inc. All rights reserved','')
    realArticles['text'][row]=realArticles['text'][row].replace('Legg til denne tweeten pÃ¥ nettstedet ditt ved Ã¥ kopiere koden nedenfor. Finn ut mer','')
    realArticles['text'][row]=realArticles['text'][row].replace('ForhÃ¥ndsvisning','')
    realArticles['text'][row]=realArticles['text'][row].replace('BEGIN VIDEO CLIP','')
    realArticles['text'][row]=realArticles['text'][row].replace('END VIDEO CLIP','')
    realArticles['text'][row]=realArticles['text'][row].replace('COMMERICAL BREAK','')
    realArticles['text'][row]=realArticles['text'][row].replace('rr','')
    realArticles['text'][row]=realArticles['text'][row].replace('Legg denne videoen til nettstedet ditt ved Ã¥ kopiere koden nedenfor. Finn ut mer','')
    realArticles['text'][row]=realArticles['text'][row].replace('Hmm, det oppnÃ¥s ikke kontakt med tjeneren. PrÃ¸ve igjen? Inkluder originaltweeten Inkluder medier', '')
    realArticles['text'][row]=realArticles['text'][row].replace('NÃ¥r du inkluderer Twitter-innhold pÃ¥ nettstedet ditt eller i appen din, samtykker du samtidig til Twitters utvikleravtale og retningslinjer for utviklere','')
    realArticles['text'][row]=realArticles['text'][row].replace('Du bruker YouTube pÃ¥','')
    realArticles['text'][row]=realArticles['text'][row].replace('Du kan endre denne innstillingen nedenfor','')
    realArticles['text'][row]=realArticles['text'][row].replace('FINDARTICLES is a CBS Interactive portal that lets you find articles about any topic, by searching in our network of news and technology sites, including CBS News, CNET, TV.com and others','')
    realArticles['text'][row]=realArticles['text'][row].replace('COPYRIGHT Â© 2005 LexisNexis, a division of Reed Elsevier Inc. All rights reserved.','')
    realArticles['text'][row]=realArticles['text'][row].replace('Please enable cookies on your web browser in order to continue','')
    realArticles['text'][row]=realArticles['text'][row].replace('Know Your Value','')
    realArticles['text'][row]=realArticles['text'][row].replace('hotline@cqrollcall.com','')
    realArticles['text'][row]=realArticles['text'][row].replace('Transcriptions/Morningside','')
    realArticles['text'][row] = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?Â«Â»â€œâ€â€˜â€™]))''', " ", realArticles['text'][row])
#remove videos by metadata
#CONTENT - conference report
fakeArticles =fakeArticles[~fakeArticles.text.str.contains(u"æ",na=False)]
fakeArticles = fakeArticles[~fakeArticles.text.str.contains(u"ø", na=False)]
fakeArticles = fakeArticles[~fakeArticles.text.str.contains(u"å",na=False)]
fakeArticles.reset_index(drop=True,inplace=True)
realArticles =realArticles[~realArticles.text.str.contains(u"æ",na=False)]
realArticles = realArticles[~realArticles.text.str.contains(u"ø", na=False)]
realArticles = realArticles[~realArticles.text.str.contains(u"å",na=False)]

fakeArticles.reset_index(drop=True,inplace=True)
#realArticles = realArticles[~realArticles.text.str.contains("rr")]
realArticles.reset_index(drop=True, inplace=True)

import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

import math
print(fakeArticles['text'].shape)
fakeArticles=fakeArticles[fakeArticles.astype(str)['movies'] == '[]']
realArticles=realArticles[realArticles.astype(str)['movies'] == '[]']
print(fakeArticles.shape)
print(realArticles.shape)
fakeIndexes=[]
import re
from nltk.corpus import words
terms = {}
fakeArticlesTermFreq = {}
realArticlesTermFreq = {}
for i, art in enumerate(fakeArticles['text']):
        #print(fakeArticlesPD['text'][row])
    try:
        terms[i] = fakeArticles['text'][i].split()
        words = str(terms[i])
    
        wordsList = words.split()
    #remove symbols
        for j,word in enumerate(wordsList):
            wordsList[j] = wordsList[j].replace("'","")
            wordsList[j] = wordsList[j].replace("\[","")
            wordsList[j] = wordsList[j].replace(":","")
            wordsList[j] = wordsList[j].replace("-","")
            wordsList[j] = wordsList[j].replace("_","")
            wordsList[j] = wordsList[j].replace("+","")
            wordsList[j] = wordsList[j].replace("?","")
            wordsList[j] = wordsList[j].replace(",","")
            wordsList[j] = wordsList[j].replace("\]","")
            wordsList[j] = wordsList[j].replace(".","")
            wordsList[j] = wordsList[j].replace("â€˜","")
            wordsList[j] = wordsList[j].replace("â€™","")
            wordsList[j] = wordsList[j].replace("@","")
            wordsList[j] = wordsList[j].replace("!","")
            wordsList[j] = wordsList[j].replace("$","")
            wordsList[j] = wordsList[j].replace("=","")
            wordsList[j] = wordsList[j].replace("(","")
            wordsList[j] = wordsList[j].replace(")","")
            wordsList[j] = wordsList[j].replace("/","")
            wordsList[j] = wordsList[j].replace("\\","")
            wordsList[j] = wordsList[j].replace("&","")
            wordsList[j] = wordsList[j].replace("[]","")
            wordsList[j] = wordsList[j].replace('"',"")
            wordsList[j] = wordsList[j].replace("â€˜â€˜","")
            wordsList[j] = wordsList[j].replace("â€™â€™","")
            wordsList[j] = re.sub('\d', '', wordsList[j])
            #remove non english words:
            
            #numberswordsList[i] = word.replace("â€œ","")


        
        wordsList = [x for x in wordsList if x not in stopwords.words('english')]
        #wordsList = [remove_symbols(i) for i in wordsList]
        uniqueWords = []
        for word in wordsList:
            #print(word)
            counts = {}
            if word not in counts.keys():
                uniqueWords.append(word)
                counts[word] = 0
            counts[word] += 1
        fakeArticlesTermFreq[i]={}
        for word in uniqueWords:
            if word not in fakeArticlesTermFreq[i].keys():
                fakeArticlesTermFreq[i][word] = 0
            for n_word in counts:
            #print(counts[n_word])
                fakeArticlesTermFreq[i][word]=counts[n_word]/len(wordsList)
    #print(fakeArticlesTermFreq[user])
    #print(fakeUsersTermFreq[user])
    except KeyError:
        fakeIndexes.append(i)
        pass
realIndexes = [] 
terms={}
for i, art in enumerate(realArticles['text']):
        #print(fakeArticlesPD['text'][row])
    try:
        terms[i] = realArticles['text'][i].split()
        words = str(terms[i])
        wordsList = words.split()
        
        #remove symbols
        for j,word in enumerate(wordsList):
            wordsList[j] = wordsList[j].replace("'","")
            wordsList[j] = wordsList[j].replace("\[","")
            wordsList[j] = wordsList[j].replace(":","")
            wordsList[j] = wordsList[j].replace("-","")
            wordsList[j] = wordsList[j].replace("_","")
            wordsList[j] = wordsList[j].replace("+","")
            wordsList[j] = wordsList[j].replace("?","")
            wordsList[j] = wordsList[j].replace(",","")
            wordsList[j] = wordsList[j].replace("\]","")
            wordsList[j] = wordsList[j].replace(".","")
            wordsList[j] = wordsList[j].replace("â€˜","")
            wordsList[j] = wordsList[j].replace("â€™","")
            wordsList[j] = wordsList[j].replace("@","")
            wordsList[j] = wordsList[j].replace("!","")
            wordsList[j] = wordsList[j].replace("$","")
            wordsList[j] = wordsList[j].replace("=","")
            wordsList[j] = wordsList[j].replace("(","")
            wordsList[j] = wordsList[j].replace(")","")
            wordsList[j] = wordsList[j].replace("/","")
            wordsList[j] = wordsList[j].replace("\\","")
            wordsList[j] = wordsList[j].replace("&","")
            wordsList[j] = wordsList[j].replace("[]","")
            wordsList[j] = wordsList[j].replace('"',"")
            wordsList[j] = wordsList[j].replace("]","")
            wordsList[j] = wordsList[j].replace("[","")
            wordsList[j] = wordsList[j].replace("â€˜â€˜","")
            wordsList[j] = wordsList[j].replace("â€™â€™","")
            wordsList[j] = re.sub('\d', '', wordsList[j])
        wordsList = [x for x in wordsList if x not in stopwords.words('english')]
        #wordsList = [remove_symbols(i) for i in wordsList]
        #uniqueWords = set(wordsList)
        #print(uniqueWords)
        uniqueWords = []
        for word in wordsList:
            #print(word)
            counts = {}
            if word not in counts.keys():
                uniqueWords.append(word)
                counts[word] = 0
            counts[word] += 1
        realArticlesTermFreq[i]={}
        for word in uniqueWords:
            if word not in realArticlesTermFreq[i].keys():
                realArticlesTermFreq[i][word] = 0
            for n_word in counts:
            #print(counts[n_word])
                realArticlesTermFreq[i][word]=counts[n_word]/len(wordsList)
    except KeyError:
        realIndexes.append(i+len(fakeArticles['text']))
        pass
print(realArticlesTermFreq[1])
import pickle
with open('fakeArticlesTermFreq.pickle', 'wb') as handle:
    pickle.dump(fakeArticlesTermFreq, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('realArticlesTermFreq.pickle', 'wb') as handle:
    pickle.dump(realArticlesTermFreq, handle, protocol=pickle.HIGHEST_PROTOCOL)



#IDF = LOG(N_art/Ni_term) - number of documents/number of documents term appear in
import math
N_art_tot = len(fakeArticles)+len(realArticles)
fakeArticlesIDF = {}
realArticlesIDF={}
counts = {}
N_art = 0
userTerms = {}


for i, art in enumerate(fakeArticles['text']):
        #print(fakeArticlesPD['text'][row])
    try:
        terms[i] = fakeArticles['text'][i].split()
        words = str(terms[i])
    
        wordsList = words.split()
        #remove symbols
        for j,word in enumerate(wordsList):
            wordsList[j] = wordsList[j].replace("'","")
            wordsList[j] = wordsList[j].replace("\[","")
            wordsList[j] = wordsList[j].replace(":","")
            wordsList[j] = wordsList[j].replace("-","")
            wordsList[j] = wordsList[j].replace("_","")
            wordsList[j] = wordsList[j].replace("+","")
            wordsList[j] = wordsList[j].replace("?","")
            wordsList[j] = wordsList[j].replace(",","")
            wordsList[j] = wordsList[j].replace("\]","")
            wordsList[j] = wordsList[j].replace(".","")
            wordsList[j] = wordsList[j].replace("â€˜","")
            wordsList[j] = wordsList[j].replace("â€™","")
            wordsList[j] = wordsList[j].replace("@","")
            wordsList[j] = wordsList[j].replace("!","")
            wordsList[j] = wordsList[j].replace("$","")
            wordsList[j] = wordsList[j].replace("=","")
            wordsList[j] = wordsList[j].replace("(","")
            wordsList[j] = wordsList[j].replace(")","")
            wordsList[j] = wordsList[j].replace("/","")
            wordsList[j] = wordsList[j].replace("\\","")
            wordsList[j] = wordsList[j].replace("&","")
            wordsList[j] = wordsList[j].replace("]","")
            wordsList[j] = wordsList[j].replace("[","")
            wordsList[j] = re.sub('\d', '', wordsList[j])
        wordsList = [x for x in wordsList if x not in stopwords.words('english')]
        #wordsList = [remove_symbols(i) for i in wordsList]
        #uniqueWords = set(wordsList)
        #print(uniqueWords)
        for word in wordsList:
            if word not in counts.keys():
                counts[word] = 0
            if counts[word] == N_art:
                counts[word] += 1
        N_art += 1      
    except KeyError:
        pass
print('check')   
       
N_art = 0
userTerms = {}
for i, art in enumerate(realArticles['text']):
        #print(fakeArticlesPD['text'][row])
    try:
        terms[i] = realArticles['text'][i].split()
        words = str(terms[i])
    
        wordsList = words.split()
        #remove symbols
        for j,word in enumerate(wordsList):
            wordsList[j] = wordsList[j].replace("'","")
            wordsList[j] = wordsList[j].replace("\[","")
            wordsList[j] = wordsList[j].replace(":","")
            wordsList[j] = wordsList[j].replace("-","")
            wordsList[j] = wordsList[j].replace("_","")
            wordsList[j] = wordsList[j].replace("+","")
            wordsList[j] = wordsList[j].replace("?","")
            wordsList[j] = wordsList[j].replace(",","")
            wordsList[j] = wordsList[j].replace("\]","")
            wordsList[j] = wordsList[j].replace(".","")
            wordsList[j] = wordsList[j].replace("@","")
            wordsList[j] = wordsList[j].replace("!","")
            wordsList[j] = wordsList[j].replace("$","")
            wordsList[j] = wordsList[j].replace("=","")
            wordsList[j] = wordsList[j].replace("(","")
            wordsList[j] = wordsList[j].replace(")","")
            wordsList[j] = wordsList[j].replace("/","")
            wordsList[j] = wordsList[j].replace("\\","")
            wordsList[j] = wordsList[j].replace("&","")
            wordsList[j] = re.sub('\d', '', wordsList[j])
        wordsList = [x for x in wordsList if x not in stopwords.words('english')]
        #wordsList = [remove_symbols(i) for i in wordsList]
        #uniqueWords = set(wordsList)
        #print(uniqueWords)
        for word in wordsList:
            if word not in counts.keys():
                counts[word] = 0
            if counts[word] == N_art:
                counts[word] += 1
        N_art += 1 
    except KeyError:
        pass
print('check')  


for i, art in enumerate(fakeArticles['text']):
    print(i)
    try:
        try:
            terms[i] = []
            uniqueWords = []
            fakeArticlesIDF[i] = {}
        #print(fakeArticlesPD['text'][row])
            terms[i] = fakeArticles['text'][i].split()
            words = str(terms[i])
            wordsList = words.split()
        except KeyError:
            if i not in fakeIndexes:
                fakeIndexes.append(i)
        #remove symbols
        for j,word in enumerate(wordsList):
            wordsList[j] = wordsList[j].replace("'","")
            wordsList[j] = wordsList[j].replace("\[","")
            wordsList[j] = wordsList[j].replace(":","")
            wordsList[j] = wordsList[j].replace("-","")
            wordsList[j] = wordsList[j].replace("_","")
            wordsList[j] = wordsList[j].replace("+","")
            wordsList[j] = wordsList[j].replace("?","")
            wordsList[j] = wordsList[j].replace(",","")
            wordsList[j] = wordsList[j].replace("\]","")
            wordsList[j] = wordsList[j].replace(".","")
            wordsList[j] = wordsList[j].replace("â€˜","")
            wordsList[j] = wordsList[j].replace("â€™","")
            wordsList[j] = wordsList[j].replace("@","")
            wordsList[j] = wordsList[j].replace("!","")
            wordsList[j] = wordsList[j].replace("$","")
            wordsList[j] = wordsList[j].replace("=","")
            wordsList[j] = wordsList[j].replace("(","")
            wordsList[j] = wordsList[j].replace(")","")
            wordsList[j] = wordsList[j].replace("/","")
            wordsList[j] = wordsList[j].replace("\\","")
            wordsList[j] = wordsList[j].replace("&","")
            wordsList[j] = re.sub('\d', '', wordsList[j])
        wordsList = [x for x in wordsList if x not in stopwords.words('english')]
        #wordsList = [remove_symbols(i) for i in wordsList]
        #uniqueWords = set(wordsList)
        #print(uniqueWords)
            #print(word)
        for word in wordsList:
            if word not in uniqueWords:
                uniqueWords.append(word)
        for word in uniqueWords:
            if word not in fakeArticlesIDF[i].keys():
                fakeArticlesIDF[i][word] = 0
            for count in counts:
            #print(counts[n_word])
                fakeArticlesIDF[i][word]=math.log(1+N_art_tot/(1+counts[word])) #+1 to avoid zero division
        #print(fakeArticlesIDF[i])
    except KeyError:
        #if i not in fakeIndexes:
        #    fakeIndexes.append(i)
        pass
print('check')  
      
for i, art in enumerate(realArticles['text']):
    print(i)
    try:
        try:
            terms[i] = []
            uniqueWords = []
            realArticlesIDF[i] = {}
        #print(fakeArticlesPD['text'][row])
            terms[i] = realArticles['text'][i].split()
            words = str(terms[i])
            wordsList = words.split()
        except KeyError:
            if (i+len(realArticles['text'])) not in realIndexes:
                realIndexes.append(i+len(realArticles['text']))
        #remove symbols
        for j,word in enumerate(wordsList):
            wordsList[j] = wordsList[j].replace("'","")
            wordsList[j] = wordsList[j].replace("\[","")
            wordsList[j] = wordsList[j].replace(":","")
            wordsList[j] = wordsList[j].replace("-","")
            wordsList[j] = wordsList[j].replace("_","")
            wordsList[j] = wordsList[j].replace("+","")
            wordsList[j] = wordsList[j].replace("?","")
            wordsList[j] = wordsList[j].replace(",","")
            wordsList[j] = wordsList[j].replace("\]","")
            wordsList[j] = wordsList[j].replace(".","")
            wordsList[j] = wordsList[j].replace("â€˜","")
            wordsList[j] = wordsList[j].replace("â€™","")
            wordsList[j] = wordsList[j].replace("@","")
            wordsList[j] = wordsList[j].replace("!","")
            wordsList[j] = wordsList[j].replace("$","")
            wordsList[j] = wordsList[j].replace("=","")
            wordsList[j] = wordsList[j].replace("(","")
            wordsList[j] = wordsList[j].replace(")","")
            wordsList[j] = wordsList[j].replace("/","")
            wordsList[j] = wordsList[j].replace("\\","")
            wordsList[j] = wordsList[j].replace("&","")
            wordsList[j] = re.sub('\d', '', wordsList[j])
        wordsList = [x for x in wordsList if x not in stopwords.words('english')]
        #wordsList = [remove_symbols(i) for i in wordsList]
        #uniqueWords = set(wordsList)
        #print(uniqueWords)
        #print(word)
        for word in wordsList:
            if word not in uniqueWords:
                uniqueWords.append(word)
        for word in uniqueWords:
            if word not in realArticlesIDF[i].keys():
                realArticlesIDF[i][word] = 0
            for count in counts:
            #print(counts[n_word])
                realArticlesIDF[i][word]=math.log(1+N_art_tot/(1+counts[word])) #+1 to avoid zero division
    except KeyError:
        #if (i+len(fakeArticles['text'])) not in realIndexes:
            #realIndexes.append(i+len(fakeArticles['text']))
        pass
    #print(fakeArticlesIDF[i])

import pickle
with open('fakeIDF.pickle', 'wb') as handle:
    pickle.dump(fakeArticlesIDF, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('realIDF.pickle', 'wb') as handle:
    pickle.dump(realArticlesIDF, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('fakeIndexes.pickle', 'wb') as handle:
    pickle.dump(fakeArticlesIDF, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('realIndexes.pickle', 'wb') as handle:
    pickle.dump(realArticlesIDF, handle, protocol=pickle.HIGHEST_PROTOCOL)

