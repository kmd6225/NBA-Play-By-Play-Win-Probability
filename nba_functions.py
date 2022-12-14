dfrom nba_api.stats.endpoints import playbyplayv2
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamestimatedmetrics
import pandas as pd
from nba_api.stats.endpoints import boxscoresummaryv2 
import numpy as np
import random

def get_team_games(team_name, year):

	#get team id

	teams_ = teams.get_teams()
	team = [x for x in teams_ if x['full_name'] == team_name][0]
	team_id = team['id']



	#get dataframe of games that the team played in during specified season

	team_games = teamgamelog.TeamGameLog(team_id = team_id, season = year).get_data_frames()[0]



	#get the game ids of the games played by specified team and convert to a list

	team_game_ids = team_games['Game_ID'].to_list()

	#get play by play data for first 14 games played by specified team

	game_plays = playbyplayv2.PlayByPlayV2(end_period = 4, start_period = 1, game_id = team_game_ids[0]).get_data_frames()[0]
	for i in list(np.linspace(1,30,num = 30, dtype = int)):
	    df1 = playbyplayv2.PlayByPlayV2(end_period = 4, start_period = 1, game_id = team_game_ids[i]).get_data_frames()[0]
	    game_plays = pd.concat([game_plays,df1], axis = 0)

	#get only desired columns 

	game1 = game_plays[game_plays.columns[range(11)]]

	return(game1)


def get_scores(game1):

	#converts None values to 0 if no baskets have been scored yet. Otherwise, converts None values to the current score

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
	            
	    
	#Split the score by the dash into two seperate columns

	scores_df = game1['SCORE'].str.split('-', expand = True)

	scores_df.columns = ['Away', 'Home']


	# Convert strings to integers


	for i in range(len(scores_df)):
	    scores_df.iloc[i,0] = int(scores_df.iloc[i,0])
	    scores_df.iloc[i,1] = int(scores_df.iloc[i,1])


	# Create new columns in game1 with the Home and Away Scores


	game1['Home_Score'] = scores_df['Home']
	game1['Away_Score'] = scores_df['Away']


	return(game1)


def get_winner(game1):

	#determine if the home team wins and populate a list with an indicator 

	Home_Team_Wins = []
	ids = np.unique(game1['GAME_ID'])
	for j in ids:
	    df3 = game1[game1['GAME_ID'] == j]
	    if max(df3['Home_Score']) > max(df3['Away_Score']):
	        Home_Team_Wins.append(1)
	    else:
	        Home_Team_Wins.append(0)
	    
	#create a dataframe of game_ids and the indicator of whether the home team won or not

	winner_df = pd.DataFrame({'Home_Team_Wins' : Home_Team_Wins, 'GAME_ID' : ids})


	#randomly sample 8 game ids out of the 14 for training the model

	train_ids = random.sample(winner_df['GAME_ID'].to_list(),8)


	game1 = pd.merge(game1, winner_df, on = 'GAME_ID')


	#get dummy variables of the play type and concatenate to the game1 df. Save as new dataframe

	event_types = pd.get_dummies(game1['EVENTMSGTYPE'])

	game1_2 = pd.concat([game1, event_types],axis = 1)

	#save new dataframe with desired columns

	gamefinal = game1_2.iloc[0:len(game1_2), [0,4,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27]]


	return(gamefinal,game1_2, train_ids)


def get_time(gamefinal):
    timedf = gamefinal['PCTIMESTRING'].str.split(':', expand = True)
    timedf.columns = ['Minutes', 'Seconds']

    for i in range(len(timedf)):
        timedf.iloc[i,0] = int(timedf.iloc[i,0])
        timedf.iloc[i,1] = int(timedf.iloc[i,1])

    gamefinal['Minutes'] = timedf['Minutes'].tolist()
    gamefinal['Seconds'] = timedf['Seconds'].tolist()

    return(gamefinal)

def get_metrics(gamefinal):
    metrics = teamestimatedmetrics.TeamEstimatedMetrics(league_id = '00',season = '2021-22',season_type  = 'Regular Season').get_data_frames()[0].iloc[0:31,[1,9]]
    home_metrics = metrics.copy()
    away_metrics = metrics.copy() 

    home_metrics = home_metrics.rename(columns = {'TEAM_ID': 'HOME_TEAM_ID', 'E_NET_RATING': 'HOME_E_NET_RATING'})
    away_metrics = away_metrics.rename(columns = {'TEAM_ID': 'VISITOR_TEAM_ID', 'E_NET_RATING': 'VISITOR_E_NET_RATING'})

    gameids = gamefinal['GAME_ID'].unique().tolist()

    home_visitor = boxscoresummaryv2.BoxScoreSummaryV2(gameids[0]).get_data_frames()[0][['GAME_ID', 'HOME_TEAM_ID','VISITOR_TEAM_ID']]
    for i in np.linspace(1,30,num = 30, dtype = int) :
        temp = boxscoresummaryv2.BoxScoreSummaryV2(gameids[i]).get_data_frames()[0]
        home_visitor = pd.concat([home_visitor, temp[['GAME_ID', 'HOME_TEAM_ID','VISITOR_TEAM_ID']]], axis = 0)

    gamefinal = gamefinal.merge(home_visitor, on = 'GAME_ID')
    gamefinal = gamefinal.merge(home_metrics, on = 'HOME_TEAM_ID')
    gamefinal = gamefinal.merge(away_metrics, on = 'VISITOR_TEAM_ID')

    gamefinal = gamefinal.drop('HOME_TEAM_ID', axis = 1)
    gamefinal = gamefinal.drop('VISITOR_TEAM_ID', axis = 1)

    return(gamefinal)







    

def get_train_test_split(gamefinal, train_ids):
    #append a boolean to list indicating if a specific game id is in train.

	istrain = []
	for i in range(len(gamefinal)):
	    if gamefinal.iloc[i,0] in train_ids:
	        istrain.append(True)
	    else:
	        istrain.append(False)


	#append a boolean to list indicating if a specific game id is in test.


	istest = []
	for i in range(len(gamefinal)):
	    if gamefinal.iloc[i,0] in train_ids:
	        istest.append(False)
	    else:
	        istest.append(True)

	#split data into train and test

	X_train = gamefinal.loc[istrain] 
	X_test = gamefinal.loc[istest]


	#get target variable


	y_train = X_train['Home_Team_Wins']
	y_test = X_test['Home_Team_Wins']


	#drop target variables from train and test to prevent data leakage


	X_train = X_train.drop('Home_Team_Wins', axis = 1)



	X_test = X_test.drop('Home_Team_Wins', axis = 1)


	#drop game_id since it isn't a predictor

	X_train = X_train.drop(['GAME_ID','HOMEDESCRIPTION','NEUTRALDESCRIPTION','VISITORDESCRIPTION','PCTIMESTRING'], axis = 1)
	X_test = X_test.drop(['GAME_ID','HOMEDESCRIPTION','NEUTRALDESCRIPTION','VISITORDESCRIPTION','PCTIMESTRING'], axis = 1)
 





	return(X_train, X_test, y_train, y_test, istest)


