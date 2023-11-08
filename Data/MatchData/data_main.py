import pandas as pd
import playerInform as pi
import numpy as np  
import getmatch as gm
if __name__ == '__main__':
    
    with open('Data/apikey.txt', 'r') as f:
        api_key = f.read()
    print("Summoner ID를 불러오는 중입니다.")
    summonerID_list = pi.getSummonerID(api_key)
    print("puuid를 불러오는 중입니다.")
    puuid_list = pi.getPuuid(summonerID_list, api_key)
    print("Match ID를 불러오는 중입니다.")
    match_id_list = pi.getMatchID(puuid_list, api_key)
    
    match_id_list_flat = [match_id for sublist in match_id_list for match_id in sublist]
    pd.DataFrame(np.array(match_id_list_flat).reshape(-1),columns=["matchid"]).to_csv('Data/MatchData/match_id_list.csv', index=False)
    
    # match_id_list = pd.read_csv('Data/MatchData/match_id_list.csv')
    match_id_list = match_id_list['matchid'].tolist()
    gm.get_full_data(match_id_list, api_key, 25)
    
    
    