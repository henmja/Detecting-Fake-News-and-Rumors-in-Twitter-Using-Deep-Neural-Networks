import numpy as np
import math
import re
import pandas as pd
from datetime import datetime
import datetime as dtime
df = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdataClean_2.pkl')
for i,row in enumerate(df['created_at_retweets'].keys()):
    if df['created_at_retweets'][row] != '[]':
        print(df['created_at_retweets'][row])
for row in df['created_at_retweets'].keys(): 
    try:
        df['created_at_retweets'][row] = str(df['created_at_retweets'][row].strip(']['))
        df['created_at_retweets'][row] = str(df['created_at_retweets'][row].replace('\'',''))
        df['created_at_retweets'][row] = df['created_at_retweets'][row].split(',')
        df['created_at_retweets'][row].sort()
    
        dates = []
        created_date = None
        for i,date in enumerate(df['created_at_retweets'][row]):


            l = date.split(' ') #split on space, + and -

            if not re.search('[a-zA-Z]', l[0]):
                del l[0]

            try:
                del l[4]
                l[0] = l[4]
                del l[4]
                l = ' '.join(l)
                date = datetime.strptime(l,'%Y %b %d %H:%M:%S') #year-month-day(number)
                dates.append(date)
            
                if i == 0:
                    created_time = str(df['created_at'][row])
                    l = created_time.split(' ') #split on space, + and -
                    l = ' '.join(l)
                    l = l.split('+')

                    del l[1]
                    l = ' '.join(l)
                    l = l.split('-')
                    l[1] = dtime.date(1900, int(l[1]), 1).strftime('%B')
                    temp = l[1]
                    l[1] = temp[0:3]
                    l = ' '.join(l)
                    created_date = datetime.strptime(l,'%Y %b %d %H:%M:%S')
                    #print('created_date')
                    #print(created_date)
                    dif = date-created_date
                    #print(dif)
                    dif_seconds = dif.total_seconds()
                    #print(dif_seconds)
                    df['created_at_retweets'][row][i] = dif_seconds/(60*60*24)
                else:
                    dif = date-created_date
                    dif_seconds = dif.total_seconds()
                    df['created_at_retweets'][row][i] = dif_seconds/(60*60*24)
            except IndexError:
                pass
    except AttributeError:
        pass

for row in df['created_at_retweets'].keys(): 
        if not re.search(r'\d', str(df['created_at_retweets'][row])):
            df['created_at_retweets'][row] = [0]

for row, xi in enumerate(df['created_at_retweets']):
    df['created_at_retweets'][row] = np.array(xi).astype(np.float)
    if len(df['created_at_retweets'][row])>1:
        print(df['created_at_retweets'][row])

print(df['created_at_retweets'])
print('retweets')
print(len(df['created_at_retweets']))
print('shape')
print(df.shape)
print('text')
print(len(df['text']))
for i,time in enumerate(df['created_at_retweets']):
    if i<500:
        print(time)
import pickle
df.to_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdata_preprocessed.pkl')
