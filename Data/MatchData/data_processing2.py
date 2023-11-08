import pandas as pd
from tqdm import tqdm
def processing(time, index):
    
    header = ['gameId', 'blueWins', 'blueTotalGolds', 'blueCurrentGolds', 'blueTotalLevel', 'blueAvgLevel', 'blueTotalMinionKills', 'blueTotalJungleMinionKills', 'blueFirstBlood', 'blueKill', 'blueDeath', 'blueAssist', 'blueWardPlaced', 'blueWardKills', 'blueFirstTower', 'blueFirstInhibitor', 'blueFirstTowerLane', 'blueTowerKills', 'blueMidTowerKills', 'blueTopTowerKills', 'blueBotTowerKills', 'blueInhibitor', 'blueFirstDragon', 'blueDragon', 'blueRiftHeralds', 'blueBaron', 'blueFirstBaron', 'redTotalGolds', 'redCurrentGolds', 'redTotalLevel', 'redAvgLevel', 'redTotalMinionKills', 'redTotalJungleMinionKills', 'redFirstBlood', 'redKill', 'redDeath', 'redAssist', 'redWardPlaced', 'redWardKills', 'redFirstTower', 'redFirstInhibitor', 'redFirstTowerLane', 'redTowerKills', 'redMidTowerKills', 'redTopTowerKills', 'redBotTowerKills', 'redInhibitor', 'redFirstDragon', 'redDragon', 'redRiftHeralds', 'redBaron', 'redFirstBaron', 'blueAirDragon', 'blueEarthDragon', 'blueWaterDragon', 'blueFireDragon', 'blueElderDragon', 'redAirDragon', 'redEarthDragon', 'redWaterDragon', 'redFireDragon', 'redElderDragon', 'blueChamps0', 'blueChamps1', 'blueChamps2', 'blueChamps3', 'blueChamps4', 'redChamps0', 'redChamps1', 'redChamps2', 'redChamps3', 'redChamps4']
    
    for j in tqdm(index):
        result = pd.DataFrame(columns=header)
        with open('Data/MatchData/TimeLine/match_data_' + str(1) + '.csv', 'r') as f:
            df = pd.read_csv(f)
            name = df.loc[j,'gameId']
        for i in range(1, time+1):
            with open('Data/MatchData/TimeLine/match_data_' + str(i) + '.csv', 'r') as f:
                df = pd.read_csv(f)
                df = df.loc[[j]]
                result = pd.concat([result, df])
        
        result.to_csv('Data/MatchData/MatchTimeLineData/'+str(name)+'.csv', index=False, header=header)


def processing2(time):
    for i in range(1, time+1):
        temp = pd.read_csv('Data/MatchData/TimeLine/match_data_' + str(i) + '.csv')
        temp = temp.drop(['gameId'], axis=1)
        temp.to_csv('Data/MatchData/TimeLine/match_data_' + str(i) + '.csv', index=False)

processing2(25)



