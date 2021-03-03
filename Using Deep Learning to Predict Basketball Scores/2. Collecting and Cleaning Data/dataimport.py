import numpy as np
import pandas as pd

teamsfile = 'https://lp-prod-resources.s3.amazonaws.com/other/deeplearningbasketballscores/Teams.csv'
gamesfile = 'https://lp-prod-resources.s3.amazonaws.com/other/deeplearningbasketballscores/Games.csv'

teams = pd.read_csv(teamsfile)

print(teams.head())