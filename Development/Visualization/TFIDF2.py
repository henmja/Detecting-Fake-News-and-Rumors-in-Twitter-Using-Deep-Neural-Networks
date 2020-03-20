import pickle
import pandas as pd

fakeArticles = pd.read_pickle("fakeArticlesPD4.pkl")
realArticles = pd.read_pickle("realArticlesPD4.pkl")

import re
for row in fakeArticles['text'].keys():
    fakeArticles['text'][row]=fakeArticles['text'][row].replace('335 SHARES SHARE THIS STORY','')
    fakeArticles['text'][row]=fakeArticles['text'][row].replace('Vi bruker informasjonskapsler for å gjøre innholdet mer personlig, skreddersy og måle annonser og tilby en tryggere opplevelse. Når du klikker eller navigerer på nettstedet, godtar du at vi henter inn informasjon på og utenfor Facebook gjennom informasjonskapsler. Finn ut mer, blant annet om tilgjengelige kontroller, her: retningslinjer for informasjonskapsler','')
    fakeArticles['text'][row]=fakeArticles['text'][row].replace('Loading','')
    fakeArticles['text'][row]=fakeArticles['text'][row].replace('Please enable cookies on your web browser in order to continue','')
    fakeArticles['text'][row] = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", fakeArticles['text'][row])


#fakeArticles['text'][row] = re.sub('http://\S+|https://\S+', '', fakeArticles['text'][row])
#fakeArticles['text'][row] = re.sub('http[s]?://\S+', '',fakeArticles['text'][row])
#fakeArticles = fakeArticles[~fakeArticles.text.str.contains("http")]
#fakeArticles = fakeArticles[~fakeArticles.text.str.contains(".com")]

print(fakeArticles['text'])

import re
for row in realArticles['text'].keys():
    realArticles['text'][row]=realArticles['text'][row].replace('Automatisk avspilling Når automatisk avspilling er på, blir en foreslått video automatisk spilt av som neste video','')
    realArticles['text'][row]=realArticles['text'][row].replace('Neste','')
    realArticles['text'][row]=realArticles['text'][row].replace('COPYRIGHT © 2005 LexisNexis, a division of Reed Elsevier Inc. All rights reserved','')
    realArticles['text'][row]=realArticles['text'][row].replace('Legg til denne tweeten på nettstedet ditt ved å kopiere koden nedenfor. Finn ut mer','')
    realArticles['text'][row]=realArticles['text'][row].replace('Forhåndsvisning','')
    realArticles['text'][row]=realArticles['text'][row].replace('BEGIN VIDEO CLIP','')
    realArticles['text'][row]=realArticles['text'][row].replace('END VIDEO CLIP','')
    realArticles['text'][row]=realArticles['text'][row].replace('COMMERICAL BREAK','')
    realArticles['text'][row]=realArticles['text'][row].replace('rr','')
    realArticles['text'][row]=realArticles['text'][row].replace('Legg denne videoen til nettstedet ditt ved å kopiere koden nedenfor. Finn ut mer','')
    realArticles['text'][row]=realArticles['text'][row].replace('Hmm, det oppnås ikke kontakt med tjeneren. Prøve igjen? Inkluder originaltweeten Inkluder medier', '')
    realArticles['text'][row]=realArticles['text'][row].replace('Når du inkluderer Twitter-innhold på nettstedet ditt eller i appen din, samtykker du samtidig til Twitters utvikleravtale og retningslinjer for utviklere','')
    realArticles['text'][row]=realArticles['text'][row].replace('Du bruker YouTube på','')
    realArticles['text'][row]=realArticles['text'][row].replace('Du kan endre denne innstillingen nedenfor','')
    realArticles['text'][row]=realArticles['text'][row].replace('FINDARTICLES is a CBS Interactive portal that lets you find articles about any topic, by searching in our network of news and technology sites, including CBS News, CNET, TV.com and others','')
    realArticles['text'][row]=realArticles['text'][row].replace('COPYRIGHT © 2005 LexisNexis, a division of Reed Elsevier Inc. All rights reserved.','')
    realArticles['text'][row]=realArticles['text'][row].replace('Please enable cookies on your web browser in order to continue','')
    realArticles['text'][row]=realArticles['text'][row].replace('Know Your Value','')
    realArticles['text'][row]=realArticles['text'][row].replace('hotline@cqrollcall.com','')
    realArticles['text'][row]=realArticles['text'][row].replace('Transcriptions/Morningside','')
    realArticles['text'][row] = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", realArticles['text'][row])
#remove videos by metadata
#CONTENT - conference report


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
            wordsList[j] = wordsList[j].replace("‘","")
            wordsList[j] = wordsList[j].replace("’","")
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
            wordsList[j] = re.sub('\d', '', wordsList[j])
            #remove non english words:
            
            #numberswordsList[i] = word.replace("“","")


        
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
        pass

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
            wordsList[j] = wordsList[j].replace("‘","")
            wordsList[j] = wordsList[j].replace("’","")
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
            wordsList[j] = wordsList[j].replace("‘","")
            wordsList[j] = wordsList[j].replace("’","")
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
            wordsList[j] = wordsList[j].replace("‘","")
            wordsList[j] = wordsList[j].replace("’","")
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
        terms[i] = []
        uniqueWords = []
        fakeArticlesIDF[i] = {}
        #print(fakeArticlesPD['text'][row])
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
            wordsList[j] = wordsList[j].replace("‘","")
            wordsList[j] = wordsList[j].replace("’","")
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
        pass
print('check')  
      
for i, art in enumerate(realArticles['text']):
    print(i)
    try:
        terms[i] = []
        uniqueWords = []
        realArticlesIDF[i] = {}
        #print(fakeArticlesPD['text'][row])
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
            wordsList[j] = wordsList[j].replace("‘","")
            wordsList[j] = wordsList[j].replace("’","")
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
        pass
    #print(fakeArticlesIDF[i])

import pickle
with open('fakeIDF.pickle', 'wb') as handle:
    pickle.dump(fakeArticlesIDF, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('realIDF.pickle', 'wb') as handle:
    pickle.dump(realArticlesIDF, handle, protocol=pickle.HIGHEST_PROTOCOL)

