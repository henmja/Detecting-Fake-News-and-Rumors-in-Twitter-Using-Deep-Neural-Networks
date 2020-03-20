import os
import glob
import pandas as pd
fakePD = pd.DataFrame()
realPD = pd.DataFrame()
fake_Texts = dict()
fake_Texts['text'] = list()
real_Texts = dict()
real_Texts['text'] = list()
for i in range(2):
    if i == 0:
        authenticity = 'fake'
    else:
        authenticity = 'real'
    rootdir = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/'+authenticity
    
    for subdir, dirs, articles in os.walk(rootdir):
        for article in dirs:
            print(article)
            temp = pd.DataFrame()
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
                except ValueError:
                    print("ValueError")

import pickle
fakePD.to_pickle("fakeArticlesPD4.pkl")
realPD.to_pickle("realArticlesPD4.pkl")