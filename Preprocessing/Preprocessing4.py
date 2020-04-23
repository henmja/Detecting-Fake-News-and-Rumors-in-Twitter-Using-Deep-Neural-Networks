import pickle
import pandas as pd
fakeArticlesPD = pd.read_pickle("../Visualization/fakeArticlesPD_3.pkl")
realArticlesPD = pd.read_pickle("../Visualization/realArticlesPD_3.pkl")
#combine PDs
bigdata = realArticlesPD.append(fakeArticlesPD, ignore_index=True)

#drop zero non-null columns:
bigdata = bigdata.dropna(axis = 1, how = 'all')

#drop rows with less than 63 non-NAN values:
bigdata = bigdata.dropna(thresh=7) #t = 0 gives 1594 entries, t = 5 gives 1592 entries, t=10 gives 793 entries

#movel label to end:
#cols = list(df.columns.values) #Make a list of all of the columns in the df
#cols.pop(cols.index('b')) #Remove b from list
#cols.pop(cols.index('x')) #Remove x from list
#df = df[cols+['b','x']]

#display head
bigdata.head()

bigdata.to_pickle("dummy.pkl")

