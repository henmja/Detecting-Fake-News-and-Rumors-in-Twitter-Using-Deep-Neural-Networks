import pandas as pd
import pickle
import numpy as np
from datetime import datetime
import datetime as dtime
from itertools import zip_longest
import re
fakeArticlesPD = pd.read_pickle('fakeTimePD.pkl')
realArticlesPD = pd.read_pickle('realTimePD.pkl')

for row in fakeArticlesPD['created_at_retweets'].keys():
    try: 
        fakeArticlesPD['created_at_retweets'][row] = str(fakeArticlesPD['created_at_retweets'][row].strip(']['))
        fakeArticlesPD['created_at_retweets'][row] = str(fakeArticlesPD['created_at_retweets'][row].replace('\'',''))
        fakeArticlesPD['created_at_retweets'][row] = fakeArticlesPD['created_at_retweets'][row].split(',')
        fakeArticlesPD['created_at_retweets'][row].sort()
        dates = []
        created_date = None
        for i,date in enumerate(fakeArticlesPD['created_at_retweets'][row]):
        #print(fakeArticlesPD['created_at_retweets'][row])

        
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
                    created_time = str(fakeArticlesPD['created_at'][row])
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
                    fakeArticlesPD['created_at_retweets'][row][i] = dif_seconds/(60*60*24)
                else:
                    dif = date-created_date
                    dif_seconds = dif.total_seconds()
                    fakeArticlesPD['created_at_retweets'][row][i] = dif_seconds/(60*60*24)
            except IndexError:
                print('IndexError')
    except AttributeError:
        pass
    except ValueError:
        pass
    except KeyError:
        pass
    except IndexError:
        pass
        #fakeArticlesPD['created_at_retweets'][row][i] = date
#for row in fakeArticlesPD['created_at_retweets'].keys(): 
    #np.cumsum(np.array(fakeArticlesPD['created_at_retweets'][row]))
    
#fakeArticlesPD.fillna(0, inplace=True)
#for record in fakeArticlesPD['created_at_retweets']:
#    print(record)


for row in fakeArticlesPD['created_at_retweets'].keys(): 
    #fakeArticlesPD['created_at_retweets'][row] = str(fakeArticlesPD['created_at_retweets'][row]).strip('][')
    if not re.search(r'\d', str(fakeArticlesPD['created_at_retweets'][row])):
        fakeArticlesPD['created_at_retweets'][row] = [0]
a = fakeArticlesPD['created_at_retweets']
for rec in a:
    print(rec)
    
    
a=[np.array(xi).astype(np.float) for xi in a]
maxLen = 0
for flws in a:
    if len(flws)>maxLen:
        maxLen = len(flws)
    #print(maxLen)
for i,flws in enumerate(a):
    #print(i)
    while len(flws)<maxLen:
        flws = np.append(a[i],np.nan)
        a[i] = flws
#a = [np.nan if x == 0 else x for x in a]
x = np.nanmean(a,axis=0)
print(x)

for row in realArticlesPD['created_at_retweets'].keys(): 
    try:
        realArticlesPD['created_at_retweets'][row] = str(realArticlesPD['created_at_retweets'][row].strip(']['))
        realArticlesPD['created_at_retweets'][row] = str(realArticlesPD['created_at_retweets'][row].replace('\'',''))
        realArticlesPD['created_at_retweets'][row] = realArticlesPD['created_at_retweets'][row].split(',')
        realArticlesPD['created_at_retweets'][row].sort()
    
        dates = []
        created_date = None
        for i,date in enumerate(realArticlesPD['created_at_retweets'][row]):
            #print(realArticlesPD['created_at_retweets'][row])


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
                    created_time = str(realArticlesPD['created_at'][row])
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
                    realArticlesPD['created_at_retweets'][row][i] = dif_seconds/(60*60*24)
                else:
                    dif = date-created_date
                    dif_seconds = dif.total_seconds()
                    realArticlesPD['created_at_retweets'][row][i] = dif_seconds/(60*60*24)
            except IndexError:
                print('IndexError')
    except AttributeError:
        pass
        #realArticlesPD['created_at_retweets'][row][i] = date
#for row in realArticlesPD['created_at_retweets'].keys(): 
    #np.cumsum(np.array(realArticlesPD['created_at_retweets'][row]))
    
#realArticlesPD.fillna(0, inplace=True)
#for record in realArticlesPD['created_at_retweets']:
#    print(record)


for row in realArticlesPD['created_at_retweets'].keys(): 
    #realArticlesPD['created_at_retweets'][row] = str(realArticlesPD['created_at_retweets'][row]).strip('][')
    if not re.search(r'\d', str(realArticlesPD['created_at_retweets'][row])):
        realArticlesPD['created_at_retweets'][row] = [0]
a = realArticlesPD['created_at_retweets']
for rec in a:
    print(rec)
    
    
a=[np.array(xi).astype(np.float) for xi in a]
maxLen = 0
for flws in a:
    if len(flws)>maxLen:
        maxLen = len(flws)
    #print(maxLen)
for i,flws in enumerate(a):
    #print(i)
    while len(flws)<maxLen:
        flws = np.append(a[i],np.nan)
        a[i] = flws
#a = [np.nan if x == 0 else x for x in a]
x2 = np.nanmean(a,axis=0)
print(x2)


#fiks x (fÃ¸rste element er minus created_at, resten er created_at_Retweets minus forrige created_at_retweets dato)
#fyll inn nans og regn ut means
#copy paste real
#fiks cdf, og followers (read retweets)
import pandas as pd
import numpy as np
from itertools import zip_longest
fakeArticlesPD = pd.read_pickle('fakeTimePD.pkl')
realArticlesPD = pd.read_pickle('realTimePD.pkl')

for row in fakeArticlesPD['followers_count_retweets'].keys(): 
    try:
        fakeArticlesPD['followers_count_retweets'][row] = fakeArticlesPD['followers_count_retweets'][row].strip('][')
    except AttributeError:
        pass
    try:
        fakeArticlesPD['followers_count_retweets'][row] = [int(s) for s in fakeArticlesPD['followers_count_retweets'][row].split(',')]
    except ValueError:
        fakeArticlesPD['followers_count_retweets'][row] = [0]
    except AttributeError:
        fakeArticlesPD['followers_count_retweets'][row] = [0]
for row in realArticlesPD['followers_count_retweets'].keys():
    try:
        realArticlesPD['followers_count_retweets'][row] = realArticlesPD['followers_count_retweets'][row].strip('][')
    except AttributeError:
        pass
    try:
        realArticlesPD['followers_count_retweets'][row] = [int(s) for s in realArticlesPD['followers_count_retweets'][row].split(',')]
    except ValueError:
        realArticlesPD['followers_count_retweets'][row] = [0]
    except AttributeError:
        realArticlesPD['followers_count_retweets'][row] = [0]

#fakeArticlesPD.fillna(0, inplace=True)
a = fakeArticlesPD['followers_count_retweets']
a=[np.array(xi) for xi in a]
print(type(a))
maxLen = 0
#set all follower lists to same length:
for flws in a:
    if len(flws)>maxLen:
        maxLen = len(flws)
    print(maxLen)
for i,flws in enumerate(a):
    print(i)
    while len(flws)<maxLen:
        flws = np.append(a[i],np.nan)
        a[i] = flws
#a = [np.nan if x == 0 else x for x in a]
y = np.nanmean(a,axis=0)
print(y)
#sums = list(zip(*a))
#zipped_list = sums[:]
#print(zipped_list)
#y = a.mean(axis=0)
#print(y.shape)
#Convert datetime

#realArticlesPD['created_at_retweets']
#realArticlesPD['followers_count_retweets']
import pandas as pd
import numpy as np
from itertools import zip_longest
realArticlesPD = pd.read_pickle('realTimePD.pkl')

for row in realArticlesPD['followers_count_retweets'].keys():
    try:
        realArticlesPD['followers_count_retweets'][row] = realArticlesPD['followers_count_retweets'][row].strip('][')
    except AttributeError:
        pass
    try:
        realArticlesPD['followers_count_retweets'][row] = [int(s) for s in realArticlesPD['followers_count_retweets'][row].split(',')]
    except ValueError:
        realArticlesPD['followers_count_retweets'][row] = [0]
    except AttributeError:
        realArticlesPD['followers_count_retweets'][row] = [0]

#realArticlesPD.fillna(0, inplace=True)
a2 = realArticlesPD['followers_count_retweets']
a2=[np.array(xi) for xi in a2]
print(a2)
maxLen = 0
#set all follower lists to same length:
for flws in a2:
    if len(flws)>maxLen:
        maxLen = len(flws)
    #print(maxLen)
for i,flws in enumerate(a2):
    #print(i)
    while len(flws)<maxLen:
        flws = np.append(a2[i],np.nan)
        a2[i] = flws
print(a2)
#a = [np.nan if x == 0 else x for x in a]
y2 = np.nanmean(a2,axis=0)
print(y2)
#sums = list(zip(*a))
#zipped_list = sums[:]
#print(zipped_list)
#y = a.mean(axis=0)
#print(y.shape)
#Convert datetime




#Combine x and y to dic
import collections

d = dict(zip(x,y))

od = collections.OrderedDict(sorted(d.items()))

#print(list(od.values()))
x_sorted = list(od)
print(x_sorted)
y_cumsum = np.cumsum(list(od.values()))
print(y_cumsum)



import collections

d2 = dict(zip(x2,y2))

od2 = collections.OrderedDict(sorted(d2.items()))

#print(list(od.values()))
x2_sorted = list(od2)
print(x2_sorted)
y2_cumsum = np.cumsum(list(od2.values()))
print(y2_cumsum)

import matplotlib.pylab as plt
fig, ax = plt.subplots(figsize=(5, 3))
fig.subplots_adjust(bottom=0.15, left=0.2)
print(type(x))
x = np.arange(len(x))
y = np.cumsum(y)
ax.plot(x, y, 'r', label='fake')

x2 = np.arange(len(x2))
y2 = np.cumsum(y2)
ax.plot(x2, y2, 'g', label='real')
#ax.plot(x2, y2, 'g', label='real')
    
ax.set_xlabel('Hops (Retweets)')
ax.set_ylabel('Followers Reached')
ax.legend()
    

plt.savefig("Hops_Followers"+".png", bbox_inches='tight')


import matplotlib.pylab as plt
fig, ax = plt.subplots(figsize=(5, 3))
fig.subplots_adjust(bottom=0.15, left=0.2)

x_sorted = np.append(x_sorted,x2_sorted[-1])
y_cumsum = np.append(y_cumsum,y_cumsum[-1])
ax.plot(x_sorted, y_cumsum, 'r', label='fake')
x2_sorted = np.append(x2_sorted,x2_sorted[-1])
y2_cumsum = np.append(y2_cumsum,y2_cumsum[-1])
ax.plot(x2_sorted, y2_cumsum, 'g', label='real')
#ax.plot(x2, y2, 'g', label='real')
    
ax.set_xlabel('Time (Seconds)')
ax.set_ylabel('Followers Reached')
ax.legend()

#plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')

plt.savefig("Time_Cascade"+".png", bbox_inches='tight')

with open('fake_retweets_time_series.pkl','wb') as f:
    pickle.dump(x_sorted, f)
with open('fake_retweets_followers_reached.pkl','wb') as f:
    pickle.dump(y_cumsum, f)
with open('real_retweets_time_series.pkl','wb') as f:
    pickle.dump(x2_sorted, f)
with open('real_retweets_followers_reached.pkl','wb') as f:
    pickle.dump(y2_cumsum, f)

