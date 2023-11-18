import pandas as pd

def slicing(timestamp, index):
    
    slicing_point = index // 5
    
    header = ['blueWins', 'blueCurrentGolds', 'blueFirstBlood', 'blueKill', 'blueDeath', 'blueAssist', 'blueWardPlaced', 'blueWardKills', 'blueFirstTower', 'blueFirstInhibitor', 'blueFirstTowerLane', 'blueTowerKills', 'blueMidTowerKills', 'blueTopTowerKills', 'blueBotTowerKills', 'blueInhibitor', 'blueFirstDragon', 'blueDragon', 'blueRiftHeralds', 'blueBaron', 'blueFirstBaron', 'redCurrentGolds', 'redFirstBlood', 'redKill', 'redDeath', 'redAssist', 'redWardPlaced', 'redWardKills', 'redFirstTower', 'redFirstInhibitor', 'redFirstTowerLane', 'redTowerKills', 'redMidTowerKills', 'redTopTowerKills', 'redBotTowerKills', 'redInhibitor', 'redFirstDragon', 'redDragon', 'redRiftHeralds', 'redBaron', 'redFirstBaron', 'blueAirDragon', 'blueEarthDragon', 'blueWaterDragon', 'blueFireDragon', 'blueElderDragon', 'redAirDragon', 'redEarthDragon', 'redWaterDragon', 'redFireDragon', 'redElderDragon', 'blueChamps0', 'blueChamps1', 'blueChamps2', 'blueChamps3', 'blueChamps4', 'redChamps0', 'redChamps1', 'redChamps2', 'redChamps3', 'redChamps4']
    
    for i in range(1, timestamp+1):
        with open('Data/MatchData/TimeLine/match_data_' + str(i) + '.csv', 'r') as f:
            df = pd.read_csv(f)
            df = df.drop(['blueTotalGolds','blueTotalLevel','blueAvgLevel', 'blueTotalMinionKills','blueTotalJungleMinionKills','redTotalGolds', 'redTotalLevel', 'redAvgLevel', 'redTotalMinionKills', 'redTotalJungleMinionKills'], axis=1)
            test_df = df.iloc[0: slicing_point]
            test_df.to_csv('Data/TestData/match_data_' + str(i) + '.csv', index=False, header= header)
            
            train_df = df.iloc[slicing_point: index+1]
            train_df.to_csv('Data/TrainData/match_data_' + str(i) + '.csv', index=False, header= header)

if __name__ == "__main__" :
    header = ['blueWins', 'blueCurrentGolds', 'blueFirstBlood', 'blueKill', 'blueDeath', 'blueAssist', 'blueWardPlaced', 'blueWardKills', 'blueFirstTower', 'blueFirstInhibitor', 'blueFirstTowerLane', 'blueTowerKills', 'blueMidTowerKills', 'blueTopTowerKills', 'blueBotTowerKills', 'blueInhibitor', 'blueFirstDragon', 'blueDragon', 'blueRiftHeralds', 'blueBaron', 'blueFirstBaron', 'redCurrentGolds', 'redFirstBlood', 'redKill', 'redDeath', 'redAssist', 'redWardPlaced', 'redWardKills', 'redFirstTower', 'redFirstInhibitor', 'redFirstTowerLane', 'redTowerKills', 'redMidTowerKills', 'redTopTowerKills', 'redBotTowerKills', 'redInhibitor', 'redFirstDragon', 'redDragon', 'redRiftHeralds', 'redBaron', 'redFirstBaron', 'blueAirDragon', 'blueEarthDragon', 'blueWaterDragon', 'blueFireDragon', 'blueElderDragon', 'redAirDragon', 'redEarthDragon', 'redWaterDragon', 'redFireDragon', 'redElderDragon', 'blueChamps0', 'blueChamps1', 'blueChamps2', 'blueChamps3', 'blueChamps4', 'redChamps0', 'redChamps1', 'redChamps2', 'redChamps3', 'redChamps4']
    print(len(header))
    slicing(25, 9475)