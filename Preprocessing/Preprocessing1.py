import pickle
import os, json
from os import path
import pandas as pd
import numpy as np
import glob
pd.set_option('display.max_columns', None)


records_per_article = []
tweets_per_article = []
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
users = list() #LAG LISTE HELLER
texts = list()
tweetUsers = list()
retweetUsers = list() 

for i in range(2):
    if i == 0:
        authenticity = 'fake'
        
        with open('tweets_per_article.pkl', 'wb') as f:
            pickle.dump(tweets_per_article, f)
    else:
        with open('tweets_per_article.pkl', 'wb') as f:
            pickle.dump(tweets_per_article, f)

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


        fakeArticlesPD.to_pickle("fakePD.pkl")
        
        authenticity = 'real'
    rootdir = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/'+authenticity
    
    for subdir, dirs, articles in os.walk(rootdir):
        for j,article in enumerate(dirs):
            print(article)
            temp = pd.DataFrame()
            path_to_json = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/' + authenticity + '/'+ article + '/tweets' 
            json_pattern = os.path.join(path_to_json,'*.json')
            file_list = glob.glob(json_pattern)
            if 'politifact' in article:
                records_per_article.append(0)
                tweets_per_article.append(0)
            for k,file in enumerate(file_list):
                if k>50:
                    continue
                try:
                    fileName = str(file.split('/')[-1])
                    fileNames.append(fileName)
                    data = pd.read_json(open(file, "r", encoding="utf8", errors="surrogateescape"), lines=True)
                    temp = temp.append(data, ignore_index = True)
                    tweets_per_article[j] += 1
                    records_per_article[j] += 1
                    if authenticity == 'fake':
                        fakeArticlesPD = fakeArticlesPD.append(data, ignore_index = True)
                    else:
                        realArticlesPD = realArticlesPD.append(data, ignore_index = True)
                    
                except ValueError:
                    pass
                except IndexError:
                    pass
            if "user" in temp.keys():
                for dic in temp["user"]:
                    if dic['id'] not in users:
                        users.append(dic['id'])
                        
            
            if "user" in temp.keys():
                for dic in temp["user"]:
                    if dic['id'] not in tweetUsers:
                        tweetUsers.append(dic['id'])
        
            temp = pd.DataFrame()        
            path_to_json = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/politifact/'+authenticity+'/'+article+'/retweets' 
            json_pattern = os.path.join(path_to_json,'*.json')
            file_list = glob.glob(json_pattern)
            
               
            for l,file in enumerate(file_list):
                try:
                    fileName = str(file.split('/')[-1])
                    idx = fileNames.index(fileName)
                    temp1 = []
                    temp2 = []
                    temp3 = []
                    temp4 = []
                    data = pd.read_json(open(file, "r", encoding="utf8", errors="surrogateescape"), lines=True)
                    for retweet in data['retweets'][0]:
                        try:
                            users.append(retweet['id'])
                        except IndexError:
                            pass
                        except KeyError:
                            pass           
                        try:
                            retweetUsers.append(retweet['id'])
                        except IndexError:
                            pass
                        except KeyError:
                            pass
                    
                    if authenticity == 'fake':
                        for retweet in data['retweets'][0]:
                            tweets_per_article[j] += 1
                            temp1.append(retweet['created_at'])
                        fakeArticlesPD['created_at_retweets'][idx] = str(temp1)
                        for retweet in data['retweets'][0]:
                            temp2.append(retweet['user']['followers_count'])
                        fakeArticlesPD['followers_count_retweets'][idx]=str(temp2)
                    else:
                        for retweet in data['retweets'][0]:
                            tweets_per_article[j] += 1
                            temp3.append(retweet['created_at'])
                        realArticlesPD['created_at_retweets'][idx] = str(temp3)
                        for retweet in data['retweets'][0]:
                            temp4.append(retweet['user']['followers_count'])
                            
                        realArticlesPD['followers_count_retweets'][idx]=str(temp4)
                except ValueError:
                    temp2 = []
                    temp4 = []
                    try:
                        for retweet in data['retweets'][0]:
                            if authenticity == 'fake':
                                temp2.append(retweet['user']['followers_count'])
                                fakeArticlesPD['followers_count_retweets'][idx]=str(temp2)
                            else:
                                temp4.append(retweet['user']['followers_count'])
                                realArticlesPD['followers_count_retweets'][idx]=str(temp4)
                                
                    except KeyError:
                        pass
                    except IndexError:
                        pass
                    pass
                except KeyError:
                    print("KeyError")
            #Total (Retweets and Tweets)
            if i == 0:
                for user in users:
                    if user not in fakeUsers.keys():
                        fakeUsers[user] = 0
                for fakeUser in users:
                    if fakeUser in fakeUsers.keys():
                        fakeUsers[fakeUser] = fakeUsers[fakeUser]+1
            elif i == 1:
                for user in users:
                    if user not in realUsers.keys():
                        realUsers[user] = 0
                for realUser in users:
                    if realUser in realUsers.keys():
                        realUsers[realUser] = realUsers[realUser]+1
            #Retweets
            if i == 0:
                for retweet in retweetUsers:
                    if retweet not in fakeRetweets.keys():
                        fakeRetweets[retweet] = 0
                for fakeRetweet in retweetUsers:
                    if fakeRetweet in fakeRetweets.keys():
                        fakeRetweets[fakeRetweet] = fakeRetweets[fakeRetweet]+1
            elif i == 1:
                for retweet in retweetUsers:
                    if retweet not in realRetweets.keys():
                        realRetweets[retweet] = 0
                for realRetweet in retweetUsers:
                    if realRetweet in realRetweets.keys():
                        realRetweets[realRetweet] = realRetweets[realRetweet]+1         
            #Tweets
            if i == 0:
                for tweet in tweetUsers:
                    if tweet not in fakeTweets.keys():
                        fakeTweets[tweet] = 0
                for fakeTweet in tweetUsers:
                    if fakeTweet in fakeTweets.keys():
                        fakeTweets[fakeTweet] = fakeTweets[fakeTweet]+1
            elif i == 1:
                for tweet in tweetUsers:
                    if tweet not in realTweets.keys():
                        realTweets[tweet] = 0
                for realTweet in tweetUsers:
                    if realTweet in realTweets:
                        realTweets[realTweet] = realTweets[realTweet]+1
import pickle
with open('tweets_per_article.pkl', 'wb') as f:
    pickle.dump(tweets_per_article, f)
with open('records_per_article.pkl', 'wb') as f:
    pickle.dump(records_per_article, f)

with open('../Visualization/fakeUsers.pickle', 'wb') as handle:
    pickle.dump(fakeUsers, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../Visualization/realUsers.pickle', 'wb') as handle:
    pickle.dump(realUsers, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('../Visualization/fakeRetweets.pickle', 'wb') as handle:
    pickle.dump(fakeRetweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../Visualization/realRetweets.pickle', 'wb') as handle:
    pickle.dump(realRetweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
with open('../Visualization/fakeTweets.pickle', 'wb') as handle:
    pickle.dump(fakeTweets, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('../Visualization/realTweets.pickle', 'wb') as handle:
    pickle.dump(realTweets, handle, protocol=pickle.HIGHEST_PROTOCOL)


fakeArticlesPD.to_pickle("../Visualization/fakePD.pkl")
realArticlesPD.to_pickle("../Visualization/realPD.pkl")


