import glob
import os
import pickle
import pandas as pd
fakeArticlesPD = pd.read_pickle("../Visualization/fakeArticlesPD_2.pkl")
video_indexes = []
video_indexes_tfidf = []
video_index = 0
video_index_tfidf = 0
idx = 0
with open('../Visualization/realArticlesDict.pickle', 'rb') as handle:
    realArticlesPD = pickle.load(handle)
realArticlesPD = pd.DataFrame(realArticlesPD)

with open('records_per_article.pkl', 'rb') as f:
    records_per_article = pickle.load(f)
print(realArticlesPD.keys())    
#change name of text to user_text
#temp = realArticlesPD.columns.values; 
#temp[realArticlesPD.columns.get_loc('text')] = 'hashtags_text';
#realArticlesPD.columns = temp

#temp = fakeArticlesPD.columns.values; 
#temp[fakeArticlesPD.columns.get_loc('text')] = 'hashtags_text';
#fakeArticlesPD.columns = temp



fakePD = pd.DataFrame()
realPD = pd.DataFrame()

realTexts = pd.Series()
fakeTexts = pd.Series()

for i in range(2):
         
    if i == 0:
        authenticity = 'fake'
    else:
        video_index = 0
        authenticity = 'real'
    rootdir = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/'+authenticity
    
    for subdir, dirs, articles in os.walk(rootdir):
        for j,article in enumerate(dirs):
            path_to_json = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/' + authenticity + '/'+ article 
            json_pattern = os.path.join(path_to_json,'*.json')
            file_list = glob.glob(json_pattern)
            for file in file_list:
                #print(file)
                try:
                    data = pd.read_json(open(file, "r", encoding="utf8", errors="surrogateescape"), lines=True)
                    if authenticity == 'fake':
                        fakePD = fakePD.append(data, ignore_index = True)
                    else:
                        realPD = realPD.append(data, ignore_index = True)
                    #print(data['text'][0])i
                    #text = data['text'][0]
                    movie = data['movies'][0]
                    if len(movie)>0:
                        video_indexes_tfidf.append(video_index_tfidf)
                        if 'politifact' in article:
                            video_index_tfidf += 1
                    if authenticity == 'fake':
                        for record in range(records_per_article[j]):
                            fakeTexts = fakeTexts.append(pd.Series(data['text'][0]), ignore_index = True)
                            if j == 0:
                                print(movie)
                                print(data['text'][0])
                            if len(movie)>0:
                                video_indexes.append(video_index)
                                
                                
                            if 'politifact' in article:
                                video_index += 1

                            #print(data['movies'][0])
                                #fakeArticlesPD['text'][i*j]=data['text']
                                #fakeArticlesPD = pd.concat(fakeArticlesPD, data)
                            #if 'politifact' in article:
                            #    video_index += 1
                    else:
                        if len(movie)>0:
                            video_indexes_tfidf.append(video_index_tfidf)
                        if 'politifact' in article:
                            video_index_tfidf += 1
                        for record in range(records_per_article[j]):
                            realTexts = realTexts.append(pd.Series(data['text'][0]), ignore_index = True)
                            #norwegian = False
                            #try:
                            #    data['text'][0].encode(encoding='utf-8').decode('ascii')
                            #except UnicodeDecodeError:
                                #norwegian = True
                            if len(movie)>0:# or norwegian:
                                video_indexes.append(video_index)
                                #video_index += 1
                                #realArticlesPD['text'][i*j]=data['text']
                                #realArticlesPD = pd.concat(realArticlesPD.append, data)
                            #if 'politifact' in article:
                                #video_index += 1
                            if 'politifact' in article: 
                                video_index += 1
                except ValueError:
                        print("ValueError")
        
print(fakeTexts)
fakeArticlesPD['text'] = fakeTexts
realArticlesPD['text'] = realTexts
#fakeArticlesPD = pd.concat([fakeArticlesPD,fakeTexts],axis=0)
#realArticlesPD = pd.concat([realArticlesPD,realTexts],axis=0)
#print(fakeTexts)
#print(fakeArticlesPD.text)
#print('length of text attribute')
#print(len(fakeArticlesPD.text))
#print(fakeArticlesPD.keys())
#print('shape')
#print(fakeArticlesPD.shape)

import pickle
with open('video_indexes.pkl', 'wb') as f:
    pickle.dump(video_indexes, f)
with open('video_indexes_tfidf.pkl', 'wb') as f:
    pickle.dump(video_indexes_tfidf, f)
fakePD.to_pickle("../Visualization/fakeArticlesPD.pkl")
realPD.to_pickle("../Visualization/realArticlesPD.pkl")
#articlesPD with article texts
fakeArticlesPD.to_pickle("../Visualization/fakeArticlesPD_3.pkl")
realArticlesPD.to_pickle("../Visualization/realArticlesPD_3.pkl")


