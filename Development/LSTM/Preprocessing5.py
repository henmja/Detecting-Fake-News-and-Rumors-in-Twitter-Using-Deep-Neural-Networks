import os, json
from os import path
import pandas as pd
import numpy as np
import glob
user_followers = pd.DataFrame()
#followers_count = []
user_following = pd.DataFrame()
#following_count = []
for i in range(2):
    if i == 0:
        folder = 'user_followers'
    else:
        folder = 'user_following'
    rootdir = '/home/prosjekt/deepnews/fakenewsnet/data/fakenewsnet/fakenewsnet_data/'+folder
    json_pattern = os.path.join(rootdir,'*.json')
    file_list = glob.glob(json_pattern)

    for file in file_list:
        #print(file)
        data = pd.read_json(open(file, "r", encoding="utf8", errors="surrogateescape"), lines=True)
        if i==0:
            user_followers = user_followers.append(data, ignore_index = True)
            #followers_count += len(user_followers.split(', '))
        else:
            user_following = user_following.append(data, ignore_index = True)
            #following_count += len(user_followers.split(', '))

            
user_followers.to_pickle("dummy_followers.pkl")
user_following.to_pickle("dummy_following.pkl")