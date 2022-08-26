from nba_api.stats.endpoints import playbyplayv2
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
import pandas as pd
import numpy as np
import random
import matplotlib
from sklearn.linear_model import LogisticRegression

from nba_functions import *


game1 = get_team_games()

game1 = get_scores()

gamefinal, game1_2 = get_winner()

X_train, X_test, y_train, y_test = get_train_test_split()


clf = LogisticRegression(max_iter = 500)

clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)


# In[43]:


preds = pd.DataFrame(preds)
preds.columns = ['Away Team Win Prob', 'Home Team Win Prob']


final = game1_2.iloc[0:len(game1_2), [0,3,4,6,7,8,9,11,12,13]]




final


final = final.iloc[X_test.index, 0:len(final.columns)]


final = final.reset_index()


final


final = pd.concat([final,preds], axis = 1)
final


final.to_csv('final.csv')