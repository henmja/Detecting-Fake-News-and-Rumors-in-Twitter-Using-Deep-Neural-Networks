import numpy as np
import pandas as pd
import pickle
with open('fakeUsers.pickle', 'rb') as handle:
    fakeUsers = pickle.load(handle)
with open('realUsers.pickle', 'rb') as handle:
    realUsers = pickle.load(handle)
    
with open('fakeRetweets.pickle', 'rb') as handle:
    fakeRetweets = pickle.load(handle)
with open('realRetweets.pickle', 'rb') as handle:
    realRetweets = pickle.load(handle)
    
with open('fakeTweets.pickle', 'rb') as handle:
    fakeTweets = pickle.load(handle)
with open('realTweets.pickle', 'rb') as handle:
    realTweets = pickle.load(handle)
    
fakeArticlesPD = pd.read_pickle("fakePD.pkl")

with open('realArticlesDict.pickle', 'rb') as handle:
    realArticlesDict = pickle.load(handle)
realArticlesPD = pd.DataFrame(realArticlesDict)

import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df = pd.read_pickle("../LSTM/bigdataClean.pkl")
df.tail()


import matplotlib.pylab as plt
realFlw = {}
fakeFlw = {}
for i,user in enumerate(df['id_str_user']):
    try:
        if (df['label'][i]=='real'):
            temp = df['followers'][i].split(', ')
            realFlw[user] = len(temp)
    except KeyError:
        print('record removed')
for i,user in enumerate(df['id_str_user']):
    try:
        if (df['label'][i]=='fake'):
            temp = df['followers'][i].split(', ')
            fakeFlw[user] = len(temp)
    except KeyError:
        print('record removed')
for user in fakeFlw:
    if user not in realFlw.keys():
        realFlw[user] = np.nan
for user in realFlw:
    if user not in fakeFlw.keys():
        fakeFlw[user] = np.nan
        #plt.scatter(realUsersTotal.items(),fakeUsersTotal.items())

#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
idx = []
for item in range(len(realFlw.keys())):
    idx.append(item)

items = []
fakeItems=[]
for key in realFlw.keys():
    items.append(realFlw[key])
for key in fakeFlw.keys():
    fakeItems.append(fakeFlw[key])
    
keys = sorted(set(list(realFlw.keys())+list(fakeFlw.keys())))

#add values from set of real and fake to realUsersTotal if 77
d = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d[x] for x in realFlw.keys()]
for i,key in enumerate(temp):
    realFlw[key] = realFlw.pop(list(d)[i])
    
d2 = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d2[x] for x in fakeFlw.keys()]
for i,key in enumerate(temp):
    fakeFlw[key] = fakeFlw.pop(list(d2)[i])
print(np.array(items)[~np.isnan(items)].mean())
print(np.array(fakeItems)[~np.isnan(np.array(fakeItems))].mean())
ax1.scatter(realFlw.keys(), items, c='g', label='real')
ax1.scatter(fakeFlw.keys(), fakeItems, c='r', label='fake')
ax1.set_xlabel('Users')
ax1.set_ylabel('Followers')
x_arr = [x for x in realFlw.keys()]
y_arr = [y for y in fakeFlw.keys()]
ax1.text(np.mean(x_arr), np.array(items)[~np.isnan(np.array(items))].mean(), 'O', size=20, color='g')
ax1.text(np.mean(y_arr),np.array(fakeItems)[~np.isnan(np.array(fakeItems))].mean(), 'X', size=20, color='r')
plt.title("Followers Frequency Scatterplot", fontsize=17)
plt.legend();
plt.savefig("followers"+".png", bbox_inches='tight')




import matplotlib.pylab as plt
realFlw = {}
fakeFlw = {}
for i,user in enumerate(df['id_str_user']):
    try:
        if (df['label'][i]=='real'):
            temp = df['followers'][i].split(', ')
            realFlw[user] = len(temp)
    except KeyError:
        print('record removed')
for i,user in enumerate(df['id_str_user']):
    try:
        if (df['label'][i]=='fake'):
            temp = df['followers'][i].split(', ')
            fakeFlw[user] = len(temp)
    except KeyError:
        print('record removed')
#for user in fakeFlw:
#    if user not in realFlw.keys():
#        realFlw[user] = np.nan
#for user in realFlw:
#    if user not in fakeFlw.keys():
#        fakeFlw[user] = np.nan

#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)

items = []
fakeItems=[]
for key in realFlw.keys():
    items.append(realFlw[key])
for key in fakeFlw.keys():
    fakeItems.append(fakeFlw[key])

x = np.sort(items)
n = x.size
y = np.arange(1, n+1) / n

ax1.scatter(x=x, y=y, c='g', label='real')
x2 = np.sort(fakeItems)
n2 = x2.size
y2 = np.arange(1, n2+1)/n
ax1.scatter(x=x2, y=y2, c='r', label='fake')
ax1.set_xlabel('Followers')
ax1.set_ylabel('Probability')
plt.title("Followers CDF", fontsize=17)
plt.legend();
plt.savefig("followers_CDF"+".png", bbox_inches='tight')


import matplotlib.pylab as plt
realFlw = {}
fakeFlw = {}
for i,user in enumerate(df['id_str_user']):
    try:
        if (df['label'][i]=='real'):
            temp = df['following'][i].split(', ')
            realFlw[user] = len(temp)
    except KeyError:
        print('record removed')
for i,user in enumerate(df['id_str_user']):
    try:
        if (df['label'][i]=='fake'):
            temp = df['following'][i].split(', ')
            fakeFlw[user] = len(temp)
    except KeyError:
        print('record removed')
for user in fakeFlw:
    if user not in realFlw.keys():
        realFlw[user] = np.nan
for user in realFlw:
    if user not in fakeFlw.keys():
        fakeFlw[user] = np.nan
        #plt.scatter(realUsersTotal.items(),fakeUsersTotal.items())

#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
idx = []
for item in range(len(realFlw.keys())):
    idx.append(item)

items = []
fakeItems=[]
for key in realFlw.keys():
    items.append(realFlw[key])
for key in fakeFlw.keys():
    fakeItems.append(fakeFlw[key])
    
keys = sorted(set(list(realFlw.keys())+list(fakeFlw.keys())))

#add values from set of real and fake to realUsersTotal if 77
d = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d[x] for x in realFlw.keys()]
for i,key in enumerate(temp):
    realFlw[key] = realFlw.pop(list(d)[i])
    
d2 = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d2[x] for x in fakeFlw.keys()]
for i,key in enumerate(temp):
    fakeFlw[key] = fakeFlw.pop(list(d2)[i])
print(np.array(items)[~np.isnan(items)].mean())
print(np.array(fakeItems)[~np.isnan(np.array(fakeItems))].mean())
ax1.scatter(realFlw.keys(), items, c='g', label='real')
ax1.scatter(fakeFlw.keys(), fakeItems, c='r', label='fake')
ax1.set_xlabel('Users')
ax1.set_ylabel('Following')
x_arr = [x for x in realFlw.keys()]
y_arr = [y for y in fakeFlw.keys()]
ax1.text(np.mean(x_arr), np.array(items)[~np.isnan(np.array(items))].mean(), 'O', size=20, color='g')
ax1.text(np.mean(y_arr),np.array(fakeItems)[~np.isnan(np.array(fakeItems))].mean(), 'X', size=20, color='r')
plt.title("Following Frequency Scatterplot", fontsize=17)
plt.legend();
plt.savefig("following"+".png", bbox_inches='tight')





import matplotlib.pylab as plt
realFlw = {}
fakeFlw = {}
for i,user in enumerate(df['id_str_user']):
    try:
        if (df['label'][i]=='real'):
            temp = df['following'][i].split(', ')
            realFlw[user] = len(temp)
    except KeyError:
        print('record removed')
for i,user in enumerate(df['id_str_user']):
    try:
        if (df['label'][i]=='fake'):
            temp = df['following'][i].split(', ')
            fakeFlw[user] = len(temp)
    except KeyError:
        print('record removed')
#for user in fakeFlw:
#    if user not in realFlw.keys():
#        realFlw[user] = np.nan
#for user in realFlw:
#    if user not in fakeFlw.keys():
#        fakeFlw[user] = np.nan

#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)

items = []
fakeItems=[]
for key in realFlw.keys():
    items.append(realFlw[key])
for key in fakeFlw.keys():
    fakeItems.append(fakeFlw[key])

x = np.sort(items)
n = x.size
y = np.arange(1, n+1) / n

ax1.scatter(x=x, y=y, c='g', label='real')
x2 = np.sort(fakeItems)
n2 = x2.size
y2 = np.arange(1, n2+1)/n
ax1.scatter(x=x2, y=y2, c='r', label='fake')
ax1.set_xlabel('Following')
ax1.set_ylabel('Probability')
plt.title("Followers CDF", fontsize=17)
plt.legend();
plt.show()
plt.savefig("following_CDF"+".png", bbox_inches='tight')

