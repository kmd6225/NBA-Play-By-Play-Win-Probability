# NBA In Game Win Probability
![](https://github.com/kmd6225/NBA-Play-By-Play-Win-Probability/blob/main/play%20by%20play%20dashboard.png?raw=true)

Play by play data and power ranking data was scraped from the NBA's database using the nba_api package for python. After carrying out data cleaning and various transformations, I split the data into a train and test set. Next, I trained a logistic regression model on the training set and subsequently made probabilistic predictions on the test set to calculate in-game win probabilities. I then visualized the play by play data for one game in the test set using tableau. This dashboard reveals how each team's probability of victory changes throughout the game. 

# How the model works

The logistic regression uses the period, how much time remains in the period, the play type, the score, and the pre game power rankings to calculate the probability of victory for each team for each play. 

# Code is fully productionalized

The code is fully productionalized in accordance with best practices for software development. All the code is contained in functions in the nba_functions.py file. These functions each do different tasks, such as importing the data, carrying out various transformations, spliting into train and test, and making predictions. The file play_by_play_prod.py imports and calls the functions from nba_functions.py. 
