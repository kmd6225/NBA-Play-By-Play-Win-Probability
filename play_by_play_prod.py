from sklearn.linear_model import LogisticRegression

from nba_functions import *


game1 = get_team_games('Boston Celtics', '2021')

game1 = get_scores(game1)

gamefinal, game1_2, train_ids = get_winner(game1)

gamefinal = get_time(gamefinal)
gamefinal = get_metrics(gamefinal)
X_train, X_test, y_train, y_test, istest = get_train_test_split(gamefinal,train_ids)


clf = LogisticRegression(max_iter = 5000)

clf.fit(X_train, y_train)
preds = clf.predict_proba(X_test)


# In[43]:


preds = pd.DataFrame(preds)
preds.columns = ['Away Team Win Prob', 'Home Team Win Prob']


final = gamefinal[istest]



final = final.reset_index()


final = pd.concat([final,preds], axis = 1)



final.to_csv('final.csv')