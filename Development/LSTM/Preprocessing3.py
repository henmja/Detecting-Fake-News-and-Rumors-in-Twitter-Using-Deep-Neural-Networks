fakeArticlesPD = pd.read_pickle("../Visualization/fakeArticlesPD_2.pkl")

with open('../Visualization/realArticlesDict.pickle', 'rb') as handle:
    realArticlesPD = pickle.load(handle)
realArticlesPD = pd.DataFrame(realArticlesPD)

with open('tweets_per_article.pkl', 'rb') as f:
    tweets_per_article = pickle.load(f)
    
#change name of text to user_text
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('text')] = 'hashtags_text';
realArticlesPD.columns = temp

temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('text')] = 'hashtags_text';
fakeArticlesPD.columns = temp



fakePD = pd.DataFrame()
realPD = pd.DataFrame()

real_Texts = dict()
real_Texts['text'] = list()
fake_Texts = dict()
fake_Texts['text'] = list()

for i in range(2):
    if i == 0:
        authenticity = 'fake'
    else:
        authenticity = 'real'
    rootdir = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/'+authenticity
    
    for subdir, dirs, articles in os.walk(rootdir):
        for article in dirs:
            temp = pd.DataFrame()
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
                except ValueError:
                    print("ValueError")
                    
                    
            for art in file_list:
                path_to_json = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/' + authenticity + '/'+ article 
                json_pattern = os.path.join(path_to_json,'*.json')
                file_list = glob.glob(json_pattern)
                for i, file in enumerate(file_list):
                    try:
                        data = pd.read_json(open(file, "r", encoding="utf8", errors="surrogateescape"), lines=True)
                        if authenticity == 'fake':
                            for tweet in range(tweets_per_article[i]):
                                fakeTexts['text'].append(data['text'][0])
                                #fakeArticlesPD['text'][i*j]=data['text']
                                #fakeArticlesPD = pd.concat(fakeArticlesPD, data)
                        else:
                            for tweet in range(tweets_per_article[i]):
                                realTexts['text'].append(data['text'][0])
                                #realArticlesPD['text'][i*j]=data['text']
                                #realArticlesPD = pd.concat(realArticlesPD.append, data)
                    except ValueError:
                        print("ValueError")
print(fakeTexts)
fakeArticlesPD = pd.concat([fakeArticlesPD,fakeTexts],axis=1)
fakeArticlesPD = pd.concat([realArticlesPD,realTexts],axis=1)

print(fakeArticlesPD.keys())
print(realArticlesPD.shape)

import pickle
#articles for visualization
fakePD.to_pickle("../Visualization/fakeArticlesPD.pkl")
realPD.to_pickle("../Visualization/realArticlesPD.pkl")
#articlesPD with article texts
fakeArticlesPD.to_pickle("../Visualization/fakeArticlesPD_2.pkl")
realArticlesPD.to_pickle("../Visualization/realArticlesPD_2.pkl")