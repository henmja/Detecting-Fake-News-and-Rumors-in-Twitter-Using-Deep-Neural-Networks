import pickle
import os, json
from os import path
import pandas as pd
import numpy as np
import glob
pd.set_option('display.max_columns', None)

fakeArticlesPD = pd.DataFrame()
fakeArticlesPD['created_at_retweets'] = np.nan
fakeArticlesPD['followers_count_retweets'] = np.nan
realArticlesPD = pd.DataFrame()
realArticlesPD['created_at_retweets'] = np.nan
realArticlesPD['followers_count_retweets'] = np.nan
fakeUsers = {}
realUsers = {}
fakeRetweets = {}
realRetweets = {}
fakeTweets = {}
realTweets = {}
fileNames = []

for i in range(2):
#for i in range(1):
    if i == 0:
        authenticity = 'fake'
        #i = 1
        #authenticity = 'real'
        #with open('../Visualization/fakeUsers.pickle', 'rb') as handle:
        #    fakeUsers = pickle.load(handle)
        #with open('../Visualization/realUsers.pickle', 'rb') as handle:
        #    realUsers = pickle.load(handle)
    
        #with open('../Visualization/fakeRetweets.pickle', 'rb') as handle:
        #    fakeRetweets = pickle.load(handle)
        #with open('../Visualization/realRetweets.pickle', 'rb') as handle:
        #    realRetweets = pickle.load(handle)
    
        #with open('../Visualization/fakeTweets.pickle', 'rb') as handle:
        #    fakeTweets = pickle.load(handle)
        #with open('../Visualization/realTweets.pickle', 'rb') as handle:
        #    realTweets = pickle.load(handle)
        #fakeArticlesPD = pd.read_pickle('../Visualization/fakePD.pkl')
        #realArticlesPD = pd.read_pickle('../Visualization/realPD.pkl')
    else:
        #CHECKPOINT

        with open('fakeUsers.pickle', 'wb') as handle:
            pickle.dump(fakeUsers, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('realUsers.pickle', 'wb') as handle:
            pickle.dump(realUsers, handle, protocol=pickle.HIGHEST_PROTOCOL)


        with open('fakeRetweets.pickle', 'wb') as handle:
            pickle.dump(fakeRetweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('realRetweets.pickle', 'wb') as handle:
            pickle.dump(realRetweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
        with open('fakeTweets.pickle', 'wb') as handle:
            pickle.dump(fakeTweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('realTweets.pickle', 'wb') as handle:
            pickle.dump(realTweets, handle, protocol=pickle.HIGHEST_PROTOCOL)


        #fakeArticlesPD.to_pickle("fakePD.pkl")
        
        authenticity = 'real'
    rootdir = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/'+authenticity
    
    for subdir, dirs, articles in os.walk(rootdir):
        for j,article in enumerate(dirs):
            #print(article)
            #print(subdir)
            temp = pd.DataFrame()
            #path_to_json = 'C:/Users/henri/FakenewsData/fakenewsnet_data/fakenewsnet_data/politifact/fake/politifact11773/tweets' 
            path_to_json = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/' + authenticity + '/'+ article + '/tweets' 
            #print(path_to_json)
            json_pattern = os.path.join(path_to_json,'*.json')
            file_list = glob.glob(json_pattern)
            #print(article)
            for k,file in enumerate(file_list):
                if k>50:
                    continue
                #print(tweets_per_article)
                try:
                    #print(file)
                    fileName = str(file.split('/')[-1])
                    if fileName not in fileNames:
                        fileNames.append(fileName)
                    data = pd.read_json(open(file, "r", encoding="utf8", errors="surrogateescape"), lines=True)
                    temp = temp.append(data, ignore_index = True)
                    #print(type(data['retweets'][0][0]))
                    #print(type(data))
                    if authenticity == 'fake':
                        fakeArticlesPD = fakeArticlesPD.append(data, ignore_index = True)
                    else:
                        realArticlesPD = realArticlesPD.append(data, ignore_index = True)
                    
                except ValueError:
                    pass
                    #print("ValueError")
                except IndexError:
                    pass
                #print(fakeArticlesPD)
        #print(temp["user"][0]['id'])
        #print(temp['retweets'][13][0]['user']['id'])
            #print(temp.head())

        #FINN UNIKE VERDIER UNDER USER I TEMP:
            users = list() #LAG LISTE HELLER
            texts = list()
            if "user" in temp.keys():
                for dic in temp["user"]:
                    if dic['id'] not in users:
                        users.append(dic['id'])
                        
            tweetUsers = list()
            texts = list()
            if "user" in temp.keys():
                for dic in temp["user"]:
                    if dic['id'] not in tweetUsers:
                        tweetUsers.append(dic['id'])
        
            temp = pd.DataFrame()        
        #path_to_json = 'C:/Users/henri/FakenewsData/fakenewsnet_data/fakenewsnet_data/politifact/fake/politifact11773/retweets' 
            path_to_json = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/'+authenticity+'/'+article+'/retweets' 
            #print(path_to_json)
            json_pattern = os.path.join(path_to_json,'*.json')
            file_list = glob.glob(json_pattern)
            
            
            for l,file in enumerate(file_list):
                fileName = str(file.split('/')[-1])
                print(file)
            #print(file)
                try:
                    idx = fileNames.index(fileName)
                    temp1 = []
                    temp2 = []
                    temp3 = []
                    temp4 = []
                    data = pd.read_json(open(file, "r", encoding="utf8", errors="surrogateescape"), lines=True)
                    #temp = temp.append(data['created_at'], ignore_index = True)
                    #try:
                        #for retweet in data['retweets'][0]:
                            #print(retweet['created_at'])
                        #for retweet in data['retweets'][0]:
                            #print(retweet['user']['followers_count'])
                        #print(data['created_at'][0])#print(data['user'][1]['followers_count'])
                    #except KeyError:
                        #print('keyerror')
                    #
                    if authenticity == 'fake':
                        #FOLLOWERS_COUNT
                        for retweet in data['retweets'][0]:
                            temp1.append(retweet['created_at'])
                        print(temp1)
                        #print(temp1)
                        fakeArticlesPD['created_at_retweets'][idx] = str(temp1)
                        print('cell created:')
                        print(fakeArticlesPD.iloc[idx]['created_at_retweets'])
                        #print(data['retweets'][0][0])
                        #print(data['retweets'][0])
                        for retweet in data['retweets'][0]:
                            print(retweet['user']['followers_count'])
                            #print(retweet['followers_count'])
                            temp2.append(retweet['user']['followers_count'])
                        print(temp2)
                        #print(temp2)
                        fakeArticlesPD['followers_count_retweets'][idx]=str(temp2)
                        print('cell value:')
                        print(fakeArticlesPD.iloc[idx]['followers_count_retweets'])
                        #print(data['retweets'][0][0]['created_at'])
                        #fakeArticlesPD = fakeArticlesPD.concatenate(data['retweets'][0][0]['created_at'], ignore_index = True)
                    else:
                        for retweet in data['retweets'][0]:
                            temp3.append(retweet['created_at'])
                        realArticlesPD['created_at_retweets'][idx] = str(temp3)
                        for retweet in data['retweets'][0]:
                            temp4.append(retweet['user']['followers_count'])
                            
                        realArticlesPD['followers_count_retweets'][idx]=str(temp4)
                        print('cell value:')
                        try:
                            print(realArticlesPD.iloc[idx]['followers_count_retweets'])
                        except IndexError:
                            pass
                        #realArticlesPD['created_at_retweets'][idx] = data['retweets'][0][0]['created_at'][idx]
                        #realArticlesPD = realArticlesPD.concatenate(data['retweets'][0][0]['created_at'], ignore_index = True)
                except ValueError:
                    temp2 = []
                    temp4 = []
                    try:
                        for retweet in data['retweets'][0]:
                            #print(retweet['user']['followers_count'])
                            if authenticity == 'fake':
                                temp2.append(retweet['user']['followers_count'])
                                fakeArticlesPD['followers_count_retweets'][idx]=str(temp2)
                                print(temp2)
                            else:
                                temp4.append(retweet['user']['followers_count'])
                                realArticlesPD['followers_count_retweets'][idx]=str(temp4)
                                
                    except KeyError:
                        pass
                    except IndexError:
                        pass
                #except IndexError:
                #    print("IndexError")
                except KeyError:
                    print("KeyError")










fakeArticlesPD[fakeArticlesPD['followers_count_retweets'].notnull()]




fakeArticlesPD.to_pickle("fakeTimePD.pkl")
realArticlesPD.to_pickle("realTimePD.pkl")

