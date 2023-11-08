import pandas as pd
import numpy as np
from tqdm import tqdm

result = pd.read_csv('Data/MatchData/TimeLine/match_data.csv')
print(result.shape)
# for i in tqdm(result.index):
#     result.loc[i, 'gameId'] = result.loc[i, 'gameId'].replace("KR_", "")

# result.to_csv('Data/MatchData/TimeLine/match_data.csv', index=False)
