#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nba_api.stats.endpoints import playbyplayv2


# In[2]:


from nba_api.stats.endpoints import teamgamelog


# In[3]:


from nba_api.stats.static import teams


# In[4]:


import pandas as pd


# In[5]:


import numpy as np


# In[6]:


import random


# In[7]:


import matplotlib


# In[8]:


from sklearn.linear_model import LogisticRegression


# In[9]:


#get boston celtics id

teams = teams.get_teams()
BOS = [x for x in teams if x['full_name'] == 'Boston Celtics'][0]
BOS_id = BOS['id']


# In[10]:


#get games that boston played in 2021
bos_games = teamgamelog.TeamGameLog(team_id = BOS_id, season = '2021').get_data_frames()[0]


# In[11]:


# get boston game_ids

bos_game_ids = bos_games['Game_ID'].to_list()


# In[13]:


game_plays = playbyplayv2.PlayByPlayV2(end_period = 4, start_period = 1, game_id = bos_game_ids[0]).get_data_frames()[0]
for i in list(np.linspace(1,14,num = 14, dtype = int)):
    df1 = playbyplayv2.PlayByPlayV2(end_period = 4, start_period = 1, game_id = bos_game_ids[i]).get_data_frames()[0]
    game_plays = pd.concat([game_plays,df1], axis = 0)


# In[14]:


game1 = game_plays[game_plays.columns[range(11)]]


# In[15]:


game1


# In[16]:


current_id = game1.iloc[0,0]

no_score = True

for i in range(len(game1)):
    
    if game1.iloc[i,10] is None and game1.iloc[i,0] == current_id and no_score == True:
        game1.iloc[i,10] = '0-0'
        
    elif game1.iloc[i,10] is None and game1.iloc[i,0] == current_id and no_score == False:
        game1.iloc[i,10] = game1.iloc[i-1,10]
        
    elif game1.iloc[i,10] is None and game1.iloc[i,0] != current_id:
        no_score = True
        current_id = game1.iloc[i,0]
        game1.iloc[i,10] = '0-0'
    
    elif game1.iloc[i,10] is not None and game1.iloc[i,0] == current_id:
        no_score = False
            
    
        


# In[19]:


scores_df = game1['SCORE'].str.split('-', expand = True)

scores_df.columns = ['Away', 'Home']


# In[20]:


for i in range(len(scores_df)):
    scores_df.iloc[i,0] = int(scores_df.iloc[i,0])
    scores_df.iloc[i,1] = int(scores_df.iloc[i,1])


# In[21]:


game1['Home_Score'] = scores_df['Home']
game1['Away_Score'] = scores_df['Away']
game1


# In[23]:


Home_Team_Wins = []
ids = np.unique(game1['GAME_ID'])
for j in ids:
    df3 = game1[game1['GAME_ID'] == j]
    if max(df3['Home_Score']) > max(df3['Away_Score']):
        Home_Team_Wins.append(1)
    else:
        Home_Team_Wins.append(0)
    


# In[24]:


winner_df = pd.DataFrame({'Home_Team_Wins' : Home_Team_Wins, 'GAME_ID' : ids})


# In[25]:


Home_Team_Wins


# In[27]:


train_ids = random.sample(winner_df['GAME_ID'].to_list(),8)


# In[28]:


train_ids


# In[29]:


game1 = pd.merge(game1, winner_df, on = 'GAME_ID')


# In[30]:


event_types = pd.get_dummies(game1['EVENTMSGTYPE'])


# In[31]:


game1_2 = pd.concat([game1, event_types],axis = 1)


# In[32]:


game1_2.columns


# In[33]:


gamefinal = game1_2.iloc[0:len(game1_2), [0,4,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]]


# In[34]:


gamefinal


# In[35]:


istrain = []
for i in range(len(gamefinal)):
    if gamefinal.iloc[i,0] in train_ids:
        istrain.append(True)
    else:
        istrain.append(False)


# In[36]:


istest = []
for i in range(len(gamefinal)):
    if gamefinal.iloc[i,0] in train_ids:
        istest.append(False)
    else:
        istest.append(True)


# In[ ]:





# In[37]:


X_train = gamefinal.loc[istrain] 
X_test = gamefinal.loc[istest]


# In[38]:


y_train = X_train['Home_Team_Wins']
y_test = X_test['Home_Team_Wins']


# In[39]:


X_train = X_train.drop('Home_Team_Wins', axis = 1)


# In[40]:


X_test = X_test.drop('Home_Team_Wins', axis = 1)


# In[41]:


X_train = X_train.drop('GAME_ID', axis = 1)
X_test = X_test.drop('GAME_ID', axis = 1)


# In[42]:


clf = LogisticRegression(max_iter = 500)

clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)


# In[43]:


preds = pd.DataFrame(preds)
preds.columns = ['Away Team Win Prob', 'Home Team Win Prob']


# In[44]:


preds


# In[46]:


final = game1_2.iloc[0:len(game1_2), [0,3,4,6,7,8,9,11,12,13]]


# In[47]:


final


# In[48]:


final = final.iloc[X_test.index, 0:len(final.columns)]


# In[49]:


final = final.reset_index()


# In[50]:


final


# In[51]:


final = pd.concat([final,preds], axis = 1)
final


# In[148]:


from sklearn.calibration import calibration_curve


# In[52]:


final.to_csv('final.csv')


# In[ ]:




