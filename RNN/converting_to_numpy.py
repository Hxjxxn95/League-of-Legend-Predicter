import pandas as pd
import numpy as np
from tqdm import tqdm

def converting(timestamp):
    header = ['blueWins', 'blueCurrentGolds', 'blueFirstBlood', 'blueKill', 'blueDeath', 'blueAssist', 'blueWardPlaced', 'blueWardKills', 'blueFirstTower', 'blueFirstInhibitor', 'blueFirstTowerLane', 'blueTowerKills', 'blueMidTowerKills', 'blueTopTowerKills', 'blueBotTowerKills', 'blueInhibitor', 'blueFirstDragon', 'blueDragon', 'blueRiftHeralds', 'blueBaron', 'blueFirstBaron', 'redCurrentGolds', 'redFirstBlood', 'redKill', 'redDeath', 'redAssist', 'redWardPlaced', 'redWardKills', 'redFirstTower', 'redFirstInhibitor', 'redFirstTowerLane', 'redTowerKills', 'redMidTowerKills', 'redTopTowerKills', 'redBotTowerKills', 'redInhibitor', 'redFirstDragon', 'redDragon', 'redRiftHeralds', 'redBaron', 'redFirstBaron', 'blueAirDragon', 'blueEarthDragon', 'blueWaterDragon', 'blueFireDragon', 'blueElderDragon', 'redAirDragon', 'redEarthDragon', 'redWaterDragon', 'redFireDragon', 'redElderDragon', 'blueChamps0', 'blueChamps1', 'blueChamps2', 'blueChamps3', 'blueChamps4', 'redChamps0', 'redChamps1', 'redChamps2', 'redChamps3', 'redChamps4']
    df = []
    
    print("Reading Test Data")
    for i in tqdm(range(1, timestamp+1)):
        df.append( pd.read_csv('Data/TestData/match_data_' + str(i) + '.csv') )
    
    index = len(df[1]['blueWins'])
    
    data_x = []
    data_y = []
    print("Converting Test Data")
    for i in tqdm(range(index)):
        
        temp = df[0].loc[[i]].copy()
        temp2 = temp.copy()
        for j in range(1, timestamp):
            
            temp = pd.concat([temp, df[j].loc[[i]]], axis=0)
        
        data_x.append(temp.loc[:, temp.columns != 'blueWins'].to_numpy())
        data_y.append(temp2['blueWins'].to_numpy())
    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    np.save('RNN/testX', data_x)
    np.save('RNN/testY', data_y)
    #------------------------------------------------------------------------------
    print("Reading Train Data")
    df = []
    for i in tqdm(range(1, timestamp+1)):
        df.append( pd.read_csv('Data/TrainData/match_data_' + str(i) + '.csv') )
    
    index = len(df[1]['blueWins'])
    
    data_x = []
    data_y = []
    
    print("Converting Train Data")
    for i in tqdm(range(index)):
        
        temp = df[0].loc[[i]].copy()
        temp2 = temp.copy()
        for j in range(1, timestamp):
            
            temp = pd.concat([temp, df[j].loc[[i]]], axis=0)
        
        data_x.append(temp.loc[:, temp.columns != 'blueWins'].to_numpy())
        data_y.append(temp2['blueWins'].to_numpy())
    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    np.save('RNN/trainX', data_x)
    np.save('RNN/trainY', data_y)

if __name__ == '__main__':
    converting(25)


        
    
        