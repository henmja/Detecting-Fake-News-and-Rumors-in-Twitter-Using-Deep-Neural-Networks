import pandas as pd
bigdata = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/dummy.pkl')
print(bigdata.keys())
exploded = bigdata.urls_description_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_urls_description_entities_user' for col in exploded.columns]
bigdata = pd.concat([bigdata.drop(columns='urls_description_entities_user'), exploded], axis=1)

temp = bigdata.columns.values; 
temp[bigdata.columns.get_loc('0_urls_description_entities_user')] = 'zero_urls_description_entities_user';
bigdata.columns = temp

exploded = bigdata.zero_urls_description_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_zero_urls_description_entities_user' for col in exploded.columns]
bigdata = pd.concat([bigdata.drop(columns='zero_urls_description_entities_user'), exploded], axis=1)

exploded = bigdata.indices_zero_urls_entities.apply(pd.Series)
exploded.columns = [str(col) + '_indices_zero_urls_entities' for col in exploded.columns]
bigdata = pd.concat([bigdata.drop(columns='indices_zero_urls_entities'), exploded], axis=1)
temp = bigdata.columns.values; 
temp[bigdata.columns.get_loc('0_indices_zero_urls_entities')] = 'zero_indices_zero_urls_entities';
temp[bigdata.columns.get_loc('1_indices_zero_urls_entities')] = 'one_indices_zero_urls_entities';
bigdata.columns = temp

exploded = bigdata.indices_zero_urls_url_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_indices_zero_urls_url_entities_user' for col in exploded.columns]
bigdata = pd.concat([bigdata.drop(columns='indices_zero_urls_url_entities_user'), exploded], axis=1)
temp = bigdata.columns.values; 
temp[bigdata.columns.get_loc('0_indices_zero_urls_url_entities_user')] = 'zero_indices_zero_urls_url_entities_user';
temp[bigdata.columns.get_loc('1_indices_zero_urls_url_entities_user')] = 'one_indices_zero_urls_url_entities_user';
bigdata.columns = temp

exploded = bigdata.indices_zero_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_indices_zero_user_mentions_entities' for col in exploded.columns]
bigdata = pd.concat([bigdata.drop(columns='indices_zero_user_mentions_entities'), exploded], axis=1)
temp = bigdata.columns.values; 
temp[bigdata.columns.get_loc('0_indices_zero_user_mentions_entities')] = 'zero_indices_zero_user_mentions_entities';
temp[bigdata.columns.get_loc('1_indices_zero_user_mentions_entities')] = 'one_indices_zero_user_mentions_entities';
bigdata.columns = temp

exploded = bigdata.symbols_entities.apply(pd.Series)
exploded.columns = [str(col) + '_symbols_entities' for col in exploded.columns]
bigdata = pd.concat([bigdata.drop(columns='symbols_entities'), exploded], axis=1)

user_followers = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/dummy_followers.pkl')
user_following = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/Preprocessing/dummy_following.pkl')
user_followers.user_id = user_followers.user_id.astype(str)
user_following.user_id = user_following.user_id.astype(str)
user_followers = user_followers.rename(columns={'user_id': 'id_str_user'})
user_following = user_following.rename(columns={'user_id': 'id_str_user'})
bigdata = pd.merge(bigdata, user_followers, on='id_str_user', how='left')
bigdata = pd.merge(bigdata, user_following, on='id_str_user', how='left')
bigdata.head()

bigdata = bigdata.rename(columns={'followers_x': 'followers'})
bigdata = bigdata.rename(columns={'followers_y': 'following'})

#move label to last column
df1 = bigdata.pop('label') # remove column b and store it in df1
bigdata['label']=df1 # add b series as a 'new' column.

bigdata.to_pickle("/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdata.pkl")

import pandas as pd
bigdata = pd.read_pickle("/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdata.pkl")

#drop zero non-null columns:
bigdata = bigdata.dropna(axis = 1, how = 'all')

#drop rows with less than 63 non-NAN values:
bigdata = bigdata.dropna(thresh=7) #t = 0 gives 1594 entries, t = 5 gives 1592 entries, t=10 gives 793 entries
bigdata.to_pickle("/local/home/henrikm/Fakenews_Classification/Preprocessing/bigdata.pkl")



