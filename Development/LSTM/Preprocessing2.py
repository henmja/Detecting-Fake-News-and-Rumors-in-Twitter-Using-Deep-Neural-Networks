import pandas as pd

fakeArticlesPD = pd.read_pickle("../Visualization/fakePD.pkl")
realArticlesPD = pd.read_pickle("../Visualization/realPD.pkl")

#change name of text to user_text
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('text')] = 'title_text';
realArticlesPD.columns = temp

temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('text')] = 'title_text';
fakeArticlesPD.columns = temp

#explode user column
exploded = realArticlesPD.user.apply(pd.Series)
exploded.columns = [str(col) + '_user' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='user'), exploded], axis=1)

exploded = realArticlesPD.entities.apply(pd.Series)
exploded.columns = [str(col) + '_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='entities'), exploded], axis=1)

exploded = realArticlesPD.quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='quoted_status'), exploded], axis=1)

exploded = realArticlesPD.entities_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_entities_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='entities_quoted_status'), exploded], axis=1)

exploded = realArticlesPD.entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_entities_user' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='entities_user'), exploded], axis=1)

exploded = realArticlesPD.description_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_description_entities_user' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='description_entities_user'), exploded], axis=1)

exploded = realArticlesPD.hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_hashtags_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='hashtags_entities'), exploded], axis=1)

#change name of 0 to zero
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('0_hashtags_entities')] = 'zero_hashtags_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.zero_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_zero_hashtags_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='zero_hashtags_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('1_hashtags_entities')] = 'one_hashtags_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.one_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_one_hashtags_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='one_hashtags_entities'), exploded], axis=1)

#change name of 2 to two
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('2_hashtags_entities')] = 'two_hashtags_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.two_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_two_hashtags_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='two_hashtags_entities'), exploded], axis=1)

#change name of 3 to three
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('3_hashtags_entities')] = 'three_hashtags_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.three_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_three_hashtags_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='three_hashtags_entities'), exploded], axis=1)

#change name of 4 to four
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('4_hashtags_entities')] = 'four_hashtags_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.four_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_four_hashtags_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='four_hashtags_entities'), exploded], axis=1)

#change name of 5 to five
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('5_hashtags_entities')] = 'five_hashtags_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.five_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_five_hashtags_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='five_hashtags_entities'), exploded], axis=1)

#explode urls_entities_quoted_status
exploded = realArticlesPD.urls_entities_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_urls_entities_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='urls_entities_quoted_status'), exploded], axis=1)

#change name of 0 to zero
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('0_urls_entities_quoted_status')] = 'zero_urls_entities_quoted_status';
realArticlesPD.columns = temp

exploded = realArticlesPD.zero_urls_entities_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_zero_urls_entities_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='zero_urls_entities_quoted_status'), exploded], axis=1)

exploded = realArticlesPD.url_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_url_entities_user' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='url_entities_user'), exploded], axis=1)

exploded = realArticlesPD.urls_url_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_urls_url_entities_user' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='urls_url_entities_user'), exploded], axis=1)

#change name of 0 to zero
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('0_urls_url_entities_user')] = 'zero_urls_url_entities_user';
realArticlesPD.columns = temp

exploded = realArticlesPD.zero_urls_url_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_zero_urls_url_entities_user' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='zero_urls_url_entities_user'), exploded], axis=1)

exploded = realArticlesPD.urls_entities.apply(pd.Series)
exploded.columns = [str(col) + '_urls_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='urls_entities'), exploded], axis=1)

#change name of 0 to zero
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('0_urls_entities')] = 'zero_urls_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.zero_urls_entities.apply(pd.Series)
exploded.columns = [str(col) + '_zero_urls_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='zero_urls_entities'), exploded], axis=1)




#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('1_urls_entities')] = 'one_urls_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.one_urls_entities.apply(pd.Series)
exploded.columns = [str(col) + '_one_urls_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='one_urls_entities'), exploded], axis=1)

exploded = realArticlesPD.user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('0_user_mentions_entities')] = 'zero_user_mentions_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.zero_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_zero_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='zero_user_mentions_entities'), exploded], axis=1)


#explode user column
exploded = fakeArticlesPD.user.apply(pd.Series)
exploded.columns = [str(col) + '_user' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='user'), exploded], axis=1)

exploded = fakeArticlesPD.entities.apply(pd.Series)
exploded.columns = [str(col) + '_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='entities'), exploded], axis=1)

exploded = fakeArticlesPD.quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_quoted_status' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='quoted_status'), exploded], axis=1)

exploded = fakeArticlesPD.entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_entities_user' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='entities_user'), exploded], axis=1)

exploded = fakeArticlesPD.description_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_description_entities_user' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='description_entities_user'), exploded], axis=1)

exploded = fakeArticlesPD.hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_hashtags_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='hashtags_entities'), exploded], axis=1)

#change name of 0 to zero
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('0_hashtags_entities')] = 'zero_hashtags_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.zero_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_zero_hashtags_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='zero_hashtags_entities'), exploded], axis=1)

#change name of 1 to zero
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('1_hashtags_entities')] = 'one_hashtags_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.one_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_one_hashtags_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='one_hashtags_entities'), exploded], axis=1)

#change name of 2 to two
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('2_hashtags_entities')] = 'two_hashtags_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.two_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_two_hashtags_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='two_hashtags_entities'), exploded], axis=1)

#change name of 3 to three
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('3_hashtags_entities')] = 'three_hashtags_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.three_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_three_hashtags_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='three_hashtags_entities'), exploded], axis=1)

#change name of 4 to four
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('4_hashtags_entities')] = 'four_hashtags_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.four_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_four_hashtags_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='four_hashtags_entities'), exploded], axis=1)

#change name of 5 to five
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('5_hashtags_entities')] = 'five_hashtags_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.five_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_five_hashtags_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='five_hashtags_entities'), exploded], axis=1)

#change name of 6 to six
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('6_hashtags_entities')] = 'six_hashtags_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.six_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_six_hashtags_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='six_hashtags_entities'), exploded], axis=1)

#change name of 7 to seven
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('7_hashtags_entities')] = 'seven_hashtags_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.seven_hashtags_entities.apply(pd.Series)
exploded.columns = [str(col) + '_seven_hashtags_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='seven_hashtags_entities'), exploded], axis=1)

exploded = fakeArticlesPD.url_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_url_entities_user' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='url_entities_user'), exploded], axis=1)

exploded = fakeArticlesPD.urls_url_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_urls_url_entities_user' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='urls_url_entities_user'), exploded], axis=1)

#change name of 0 to zero
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('0_urls_url_entities_user')] = 'zero_urls_url_entities_user';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.zero_urls_url_entities_user.apply(pd.Series)
exploded.columns = [str(col) + '_zero_urls_url_entities_user' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='zero_urls_url_entities_user'), exploded], axis=1)

exploded = fakeArticlesPD.urls_entities.apply(pd.Series)
exploded.columns = [str(col) + '_urls_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='urls_entities'), exploded], axis=1)

#change name of 0 to zero
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('0_urls_entities')] = 'zero_urls_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.zero_urls_entities.apply(pd.Series)
exploded.columns = [str(col) + '_zero_urls_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='zero_urls_entities'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('1_urls_entities')] = 'one_urls_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.one_urls_entities.apply(pd.Series)
exploded.columns = [str(col) + '_one_urls_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='one_urls_entities'), exploded], axis=1)

exploded = fakeArticlesPD.user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_user_mentions_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('0_user_mentions_entities')] = 'zero_user_mentions_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.zero_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_zero_user_mentions_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='zero_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('1_user_mentions_entities')] = 'one_user_mentions_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.one_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_one_user_mentions_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='one_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('1_user_mentions_entities')] = 'one_user_mentions_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.one_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_one_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='one_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('2_user_mentions_entities')] = 'two_user_mentions_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.two_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_two_user_mentions_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='two_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('2_user_mentions_entities')] = 'two_user_mentions_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.two_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_two_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='two_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('3_user_mentions_entities')] = 'three_user_mentions_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.three_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_three_user_mentions_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='three_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('3_user_mentions_entities')] = 'three_user_mentions_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.three_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_three_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='three_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('4_user_mentions_entities')] = 'four_user_mentions_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.four_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_four_user_mentions_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='four_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('4_user_mentions_entities')] = 'four_user_mentions_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.four_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_four_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='four_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('5_user_mentions_entities')] = 'five_user_mentions_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.five_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_five_user_mentions_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='five_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('5_user_mentions_entities')] = 'five_user_mentions_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.five_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_five_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='five_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('6_user_mentions_entities')] = 'six_user_mentions_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.six_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_six_user_mentions_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='six_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('6_user_mentions_entities')] = 'six_user_mentions_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.six_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_six_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='six_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('7_user_mentions_entities')] = 'seven_user_mentions_entities';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.seven_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_seven_user_mentions_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='seven_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('7_user_mentions_entities')] = 'seven_user_mentions_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.seven_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_seven_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='seven_user_mentions_entities'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('8_user_mentions_entities')] = 'eight_user_mentions_entities';
realArticlesPD.columns = temp

exploded = realArticlesPD.eight_user_mentions_entities.apply(pd.Series)
exploded.columns = [str(col) + '_eight_user_mentions_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='eight_user_mentions_entities'), exploded], axis=1)

exploded = realArticlesPD.user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_user_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='user_quoted_status'), exploded], axis=1)

exploded = fakeArticlesPD.user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_user_quoted_status' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='user_quoted_status'), exploded], axis=1)

exploded = realArticlesPD.extended_entities.apply(pd.Series)
exploded.columns = [str(col) + '_extended_entities' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='extended_entities'), exploded], axis=1)

exploded = fakeArticlesPD.extended_entities.apply(pd.Series)
exploded.columns = [str(col) + '_extended_entities' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='extended_entities'), exploded], axis=1)

exploded = realArticlesPD.entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_entities_user_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='entities_user_quoted_status'), exploded], axis=1)

exploded = fakeArticlesPD.entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_entities_user_quoted_status' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='entities_user_quoted_status'), exploded], axis=1)

exploded = realArticlesPD.description_entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_description_entities_user_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='description_entities_user_quoted_status'), exploded], axis=1)

exploded = fakeArticlesPD.description_entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_description_entities_user_quoted_status' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='description_entities_user_quoted_status'), exploded], axis=1)

exploded = realArticlesPD.url_entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_url_entities_user_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='url_entities_user_quoted_status'), exploded], axis=1)

exploded = fakeArticlesPD.url_entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_url_entities_user_quoted_status' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='url_entities_user_quoted_status'), exploded], axis=1)

exploded = realArticlesPD.urls_url_entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_urls_url_entities_user_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='urls_url_entities_user_quoted_status'), exploded], axis=1)

exploded = fakeArticlesPD.urls_url_entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_urls_url_entities_user_quoted_status' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='urls_url_entities_user_quoted_status'), exploded], axis=1)

#change name of 1 to one
temp = realArticlesPD.columns.values; 
temp[realArticlesPD.columns.get_loc('0_urls__url_entities_user_quoted_status')] = 'zero_urls__url_entities_user_quoted_status';
realArticlesPD.columns = temp

exploded = realArticlesPD.zero_urls__url_entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_zero_urls_url_entities_user_quoted_status' for col in exploded.columns]
realArticlesPD = pd.concat([realArticlesPD.drop(columns='zero_urls_url_entities_user_quoted_status'), exploded], axis=1)

#change name of 1 to one
temp = fakeArticlesPD.columns.values; 
temp[fakeArticlesPD.columns.get_loc('0_urls_url_entities_user_quoted_status')] = 'zero_urls_url_entities_user_quoted_status';
fakeArticlesPD.columns = temp

exploded = fakeArticlesPD.zero_urls_url_entities_user_quoted_status.apply(pd.Series)
exploded.columns = [str(col) + '_zero_urls_url_entities_user_quoted_status' for col in exploded.columns]
fakeArticlesPD = pd.concat([fakeArticlesPD.drop(columns='zero_urls_url_entities_user_quoted_status'), exploded], axis=1)


#0_urls_url_entities_user_quoted_status

#add label to data
realArticlesPD['label'] = 'real'
#add label to data
fakeArticlesPD['label'] = 'fake'




fakeArticlesPD.to_pickle("../Visualization/fakeArticlesPD_2.pkl")
realArticlesDict = realArticlesPD.to_dict('realArticlesDict')
with open('../Visualization/realArticlesDict.pickle', 'wb') as handle:
    pickle.dump(realArticlesDict, handle, protocol=pickle.HIGHEST_PROTOCOL)