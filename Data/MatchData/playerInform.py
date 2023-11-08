import requests
import time
import errorDetect as ed

# game match 불러오기 위해 먼저 summonerID를 불러온다.
def getSummonerID(api_key):
    url = f"https://kr.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5?api_key={api_key}"
    response = requests.get(url)
    ed.errorDetect(response, url)
    
    return [entry['summonerName'] for entry in response.json()['entries']]


def getPuuid(summoner_name_list : list[str], api_key):
    puuid_list = []
    for summoner_name in summoner_name_list:
        url = f"https://kr.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summoner_name}?api_key={api_key}"
        response = requests.get(url)
        ed.errorDetect(response, url)
        
        while response.status_code == 429:
            print("API 이용량 제한. 30초 대기")
            time.sleep(30)
            response = requests.get(url)
        
        if response.status_code == 200:
            puuid_list.append(response.json().get('puuid'))
            
    return puuid_list


# game match ID 불러오기

def getMatchID( puuid_list: list[str], api_key):
    match_id_list = []
    for puuid in puuid_list:
        url = f"https://asia.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=100&api_key={api_key}"
        
        response = requests.get(url)
        ed.errorDetect(response, url)
        
        while response.status_code == 429:
            print("API 이용량 제한. 30초 대기")
            time.sleep(30)
            response = requests.get(url)
            
        if response.status_code == 200:
            match_id_list.append(response.json())
            
        
        
    return match_id_list





