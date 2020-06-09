import glob
import os
import pickle
import pandas as pd
fakeArticlesPD = pd.read_pickle("/local/home/henrikm/Fakenews_Classification/Visualization/fakeArticlesPD_2.pkl")
video_indexes = []
video_indexes_tfidf = []
video_index = 0
video_index_tfidf = 0
idx = 0
with open('/local/home/henrikm/Fakenews_Classification/Visualization/realArticlesDict.pickle', 'rb') as handle:
    realArticlesPD = pickle.load(handle)
realArticlesPD = pd.DataFrame(realArticlesPD)

with open('/local/home/henrikm/Fakenews_Classification/Preprocessing/records_per_article.pkl', 'rb') as f:
    records_per_article = pickle.load(f)
print(realArticlesPD.keys())    



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
                try:
                    data = pd.read_json(open(file, "r", encoding="utf8", errors="surrogateescape"), lines=True)
                    if authenticity == 'fake':
                        fakePD = fakePD.append(data, ignore_index = True)
                    else:
                        realPD = realPD.append(data, ignore_index = True)
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

                    else:
                        if len(movie)>0:
                            video_indexes_tfidf.append(video_index_tfidf)
                        if 'politifact' in article:
                            video_index_tfidf += 1
                        for record in range(records_per_article[j]):
                            realTexts = realTexts.append(pd.Series(data['text'][0]), ignore_index = True)
                            if len(movie)>0:# or norwegian:
                                video_indexes.append(video_index)
                            if 'politifact' in article: 
                                video_index += 1
                except ValueError:
                        print("ValueError")
        
print(fakeTexts)
fakeArticlesPD['text'] = fakeTexts
realArticlesPD['text'] = realTexts


import pickle
with open('/local/home/henrikm/Fakenews_Classification/Preprocessing/video_indexes.pkl', 'wb') as f:
    pickle.dump(video_indexes, f)
with open('/local/home/henrikm/Fakenews_Classification/Preprocessing/video_indexes_tfidf.pkl', 'wb') as f:
    pickle.dump(video_indexes_tfidf, f)
fakePD.to_pickle("/local/home/henrikm/Fakenews_Classification/Visualization/fakeArticlesPD.pkl")
realPD.to_pickle("/local/home/henrikm/Fakenews_Classification/Visualization/realArticlesPD.pkl")
fakeArticlesPD.to_pickle("/local/home/henrikm/Fakenews_Classification/Visualization/fakeArticlesPD_3.pkl")
realArticlesPD.to_pickle("/local/home/henrikm/Fakenews_Classification/Visualization/realArticlesPD_3.pkl")


