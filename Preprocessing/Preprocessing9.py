import numpy as np
import math
import re
import pandas as pd
from datetime import datetime
import datetime as dtime
df = pd.read_pickle('bigdataClean_2.pkl')
#df['created_at_retweets'].fillna(0, inplace=True)
for i,row in enumerate(df['created_at_retweets'].keys()):
    #if record is not 0
    if df['created_at_retweets'][row] != '[]':
        print(df['created_at_retweets'][row])
#for i,b in enumerate(df['created_at_retweets']):
#    if i < 1000:
#        print(b)
#        print(len(b))
for row in df['created_at_retweets'].keys(): 
    try:
        df['created_at_retweets'][row] = str(df['created_at_retweets'][row].strip(']['))
        df['created_at_retweets'][row] = str(df['created_at_retweets'][row].replace('\'',''))
        df['created_at_retweets'][row] = df['created_at_retweets'][row].split(',')
        df['created_at_retweets'][row].sort()
    
        dates = []
        created_date = None
        for i,date in enumerate(df['created_at_retweets'][row]):
            #print(df['created_at_retweets'][row])


            l = date.split(' ') #split on space, + and -

            if not re.search('[a-zA-Z]', l[0]):
                del l[0]


            #print(l)
            try:
                del l[4]
                l[0] = l[4]
                del l[4]
                l = ' '.join(l)
                date = datetime.strptime(l,'%Y %b %d %H:%M:%S') #year-month-day(number)
                dates.append(date)
            
            #print('date')
            #print(date)
                if i == 0:
                    created_time = str(df['created_at'][row])
                    l = created_time.split(' ') #split on space, + and -
                    l = ' '.join(l)
                    l = l.split('+')
        #print(l)

                    del l[1]
                    l = ' '.join(l)
                    l = l.split('-')
                    l[1] = dtime.date(1900, int(l[1]), 1).strftime('%B')
                    temp = l[1]
                    l[1] = temp[0:3]
                    l = ' '.join(l)
            #print(l)
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
            #print('IndexError')
    except AttributeError:
        pass

for row in df['created_at_retweets'].keys(): 
        #realArticlesPD['created_at_retweets'][row] = str(realArticlesPD['created_at_retweets'][row]).strip('][')
        if not re.search(r'\d', str(df['created_at_retweets'][row])):
            df['created_at_retweets'][row] = [0]

for row, xi in enumerate(df['created_at_retweets']):
    df['created_at_retweets'][row] = np.array(xi).astype(np.float)
    if len(df['created_at_retweets'][row])>1:
        print(df['created_at_retweets'][row])

#a = df['created_at_retweets']
#a=[np.array(xi).astype(np.float) for xi in df['created_at_retweets']]
#print(len(a))
#df['created_at_retweets'] = pd.DataFrame(a)
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
df.to_pickle('bigdata_timeseries.pkl')
#for i,a in enumerate(df['created_at_retweets']):
    #if i<1000 and not math.isnan(a):
        #print(a)
    #if a is not 'NaN' and a and i<1000:
    #    print(a)
#print(df['created_at_retweets'])
