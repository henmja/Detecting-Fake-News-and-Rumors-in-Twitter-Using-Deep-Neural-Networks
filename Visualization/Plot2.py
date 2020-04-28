import numpy as np
import pandas as pd
import pickle
with open('fakeUsers.pickle', 'rb') as handle:
    fakeUsers = pickle.load(handle)
with open('realUsers.pickle', 'rb') as handle:
    realUsers = pickle.load(handle)
print(fakeUsers)
print(realUsers)
with open('fakeRetweets.pickle', 'rb') as handle:
    fakeRetweets = pickle.load(handle)
with open('realRetweets.pickle', 'rb') as handle:
    realRetweets = pickle.load(handle)
print(fakeRetweets)
print(realRetweets)
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
df = pd.read_pickle("../Preprocessing/bigdataClean.pkl")
df.tail()


#Ratio-Plot
import matplotlib.pylab as plt
realUsersTotal = {}
fakeUsersTotal = {}
realUsersTotal = dict(realUsers)
fakeUsersTotal = dict(fakeUsers)

for key in fakeUsers:
    if key not in realUsers.keys():
        realUsersTotal[key] = np.nan
for key in realUsers:
    if key not in fakeUsers.keys():
        fakeUsersTotal[key] = np.nan

#plt.scatter(realUsersTotal.items(),fakeUsersTotal.items())

#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
idx = []
for item in range(len(realUsersTotal.keys())):
    idx.append(item)
items = []
fakeItems=[]
for key in realUsersTotal.keys():
    items.append(realUsersTotal[key])
for key in fakeUsersTotal.keys():
    fakeItems.append(fakeUsersTotal[key])
    
keys = sorted(set(list(realUsersTotal.keys())+list(fakeUsersTotal.keys())))

#add values from set of real and fake to realUsersTotal if 77
d = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d[x] for x in realUsersTotal.keys()]
for i,key in enumerate(temp):
    realUsersTotal[key] = realUsersTotal.pop(list(d)[i])
    
d2 = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d2[x] for x in fakeUsersTotal.keys()]
for i,key in enumerate(temp):
    fakeUsersTotal[key] = fakeUsersTotal.pop(list(d2)[i])

ax1.scatter(realUsersTotal.keys(), items, c='g', label='real')
ax1.scatter(fakeUsersTotal.keys(), fakeItems, c='r', label='fake')
ax1.set_xlabel('Users')
ax1.set_ylabel('Tweet/Retweet Frequency')
x_arr = [x for x in realUsersTotal.keys()]
y_arr = [y for y in fakeUsersTotal.keys()]
print(np.array(items)[~np.isnan(items)].mean())
print(np.array(fakeItems)[~np.isnan(fakeItems)].mean())
print(len(items))
print(len(fakeItems))
ax1.text(np.mean(x_arr), np.array(items)[~np.isnan(items)].mean(), 'O', size=20, color='g')
ax1.text(np.mean(y_arr),np.array(fakeItems)[~np.isnan(fakeItems)].mean(), 'X', size=20, color='r')
plt.title("Tweet/Retweet Frequency Scatterplot", fontsize=17)
plt.legend();
plt.savefig("tweet_retweet_frequency"+".png", bbox_inches='tight')



#Ratio-Plot
import matplotlib.pylab as plt
realUsersTotal = {}
fakeUsersTotal = {}
realUsersTotal = dict(realTweets)
fakeUsersTotal = dict(fakeTweets)
for key in fakeTweets:
    if key not in realTweets.keys():
        realUsersTotal[key] = np.nan
for key in realTweets:
    if key not in fakeTweets.keys():
        fakeUsersTotal[key] = np.nan

#plt.scatter(realUsersTotal.items(),fakeUsersTotal.items())

#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
idx = []
for item in range(len(realUsersTotal.keys())):
    idx.append(item)
items = []
fakeItems=[]
for key in realUsersTotal.keys():
    items.append(realUsersTotal[key])
for key in fakeUsersTotal.keys():
    fakeItems.append(fakeUsersTotal[key])
    
keys = sorted(set(list(realUsersTotal.keys())+list(fakeUsersTotal.keys())))

#add values from set of real and fake to realUsersTotal if 77
d = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d[x] for x in realUsersTotal.keys()]
for i,key in enumerate(temp):
    realUsersTotal[key] = realUsersTotal.pop(list(d)[i])
    
d2 = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d2[x] for x in fakeUsersTotal.keys()]
for i,key in enumerate(temp):
    fakeUsersTotal[key] = fakeUsersTotal.pop(list(d2)[i])
    
ax1.scatter(realUsersTotal.keys(), items, c='g', label='real')
ax1.scatter(fakeUsersTotal.keys(), fakeItems, c='r', label='fake')
x_arr = [x for x in realUsersTotal.keys()]
y_arr = [y for y in fakeUsersTotal.keys()]
print(np.array(items)[~np.isnan(items)].mean())
print(np.array(fakeItems)[~np.isnan(fakeItems)].mean())
print(len(items))
print(len(fakeItems))
ax1.text(np.mean(x_arr), np.array(items)[~np.isnan(items)].mean(), 'O', size=20, color='g')
ax1.text(np.mean(y_arr),np.array(fakeItems)[~np.isnan(fakeItems)].mean(), 'X', size=20, color='r')
ax1.set_xlabel('Users')
ax1.set_ylabel('Tweet Frequency')
plt.title("Tweet Frequency Scatterplot", fontsize=17)
plt.legend();
plt.savefig("tweet_frequency"+".png", bbox_inches='tight')





import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

realTemp=list(realUsers.values())
ax = plt.subplot(111)
x = np.sort(realTemp)
n = x.size
y = np.arange(1, n+1) / n

ax.scatter(x=x, y=y);

fakeTemp=list(fakeUsers.values())
x = np.sort(fakeTemp)
n = x.size
y = np.arange(1, n+1) / n

ax.scatter(x=x, y=y);

ax.legend(('Real','Fake'))
ax.set_xlabel('Tweet Frequency')
ax.set_ylabel('Probability')
plt.savefig("tweet CDF"+".png", bbox_inches='tight')




#Ratio-Plot
import matplotlib.pylab as plt
realUsersTotal = {}
fakeUsersTotal = {}
realUsersTotal = dict(realRetweets)
fakeUsersTotal = dict(fakeRetweets)
print(len(realUsersTotal.keys()))
for key in fakeRetweets:
    if key not in realRetweets.keys():
        realUsersTotal[key] = np.nan
for key in realRetweets:
    if key not in fakeRetweets.keys():
        fakeUsersTotal[key] = np.nan
print(len(realUsersTotal.keys()))

#plt.scatter(realUsersTotal.items(),fakeUsersTotal.items())

#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
idx = []
for item in range(len(realUsersTotal.keys())):
    idx.append(item)
print(len(idx))
items = []
fakeItems=[]
for key in realUsersTotal.keys():
    items.append(realUsersTotal[key])
for key in fakeUsersTotal.keys():
    fakeItems.append(fakeUsersTotal[key])
    
keys = sorted(set(list(realUsersTotal.keys())+list(fakeUsersTotal.keys())))

#add values from set of real and fake to realUsersTotal if 77
d = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d[x] for x in realUsersTotal.keys()]
for i,key in enumerate(temp):
    realUsersTotal[key] = realUsersTotal.pop(list(d)[i])
    
d2 = dict([(y,x+1) for x,y in enumerate(keys)])
temp = [d2[x] for x in fakeUsersTotal.keys()]
for i,key in enumerate(temp):
    fakeUsersTotal[key] = fakeUsersTotal.pop(list(d2)[i])
    
print(len(fakeUsersTotal))
ax1.scatter(realUsersTotal.keys(), items, c='g', label='real')
ax1.scatter(fakeUsersTotal.keys(), fakeItems, c='r', label='fake')
x_arr = [x for x in realUsersTotal.keys()]
y_arr = [y for y in fakeUsersTotal.keys()]
print(np.array(items)[~np.isnan(items)].mean())
print(np.array(fakeItems)[~np.isnan(fakeItems)].mean())
ax1.text(np.mean(x_arr), np.array(items)[~np.isnan(items)].mean(), 'O', size=20, color='g')
ax1.text(np.mean(y_arr),np.array(fakeItems)[~np.isnan(fakeItems)].mean(), 'X', size=20, color='r')
ax1.set_xlabel('Users')
ax1.set_ylabel('Retweet Frequency')
plt.title("Retweet Frequency Scatterplot", fontsize=17)
plt.legend();
plt.show()
plt.savefig("retweet_frequency"+".png", bbox_inches='tight')








#Retweet CDF
import matplotlib.pylab as plt
realUsersTotal = {}
fakeUsersTotal = {}
realUsersTotal = dict(realRetweets)
fakeUsersTotal = dict(fakeRetweets)
print(len(realUsersTotal.keys()))
for key in fakeRetweets:
    if key not in realRetweets.keys():
        realUsersTotal[key] = np.nan
for key in realRetweets:
    if key not in fakeRetweets.keys():
        fakeUsersTotal[key] = np.nan
print(len(realUsersTotal.keys()))

#plt.scatter(realUsersTotal.items(),fakeUsersTotal.items())

#plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
#idx = []
#for item in range(len(realUsersTotal.keys())):
#    idx.append(item)
#print(len(idx))
#items = []
#fakeItems=[]
#for key in realUsersTotal.keys():
#    items.append(realUsersTotal[key])
#for key in fakeUsersTotal.keys():
#    fakeItems.append(fakeUsersTotal[key])
print(realRetweets)
print(fakeRetweets)

x= [item for item in realUsersTotal.values()]
n = len(x)
y = np.arange(1, n+1) / n
#in_hist = [list(in_degrees.values()).count(x) for x in in_values]
#print(x.shape)
print(x)
print(type(x))
ax1.scatter(x=x, y=y, c='g', label='real')

x=[item for item in fakeUsersTotal.values()]
n = len(x)
y = np.arange(1, n+1) / n
ax1.scatter(x=x, y=y, c='r', label='fake')

#print(np.array(realUsersTotal.values())[~np.isnan(realUsersTotal.values())].mean())
#print(np.array(fakeUsersTotal.values())[~np.isnan(fakeUsersTotal.values())].mean())
ax1.set_xlabel('Retweet Frequency')
ax1.set_ylabel('Probabilities')
plt.title("Retweet Frequency CDF", fontsize=17)
plt.legend();
plt.xlim(750,1500)
plt.savefig("retweet_CDF"+".png", bbox_inches='tight')

