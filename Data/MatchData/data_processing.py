import pandas as pd
import numpy as np
from tqdm import tqdm

def processing(result):

    result = result.drop(['redWins'], axis=1)
    result = result.replace({'blueWins': True}, {'blueWins': 1})
    result = result.replace({'blueWins': False}, {'blueWins': 0})
    result = result.replace({'blueFirstTowerLane': "[\'TOP_LANE\']"}, {'blueFirstTowerLane': 1})
    result = result.replace({'blueFirstTowerLane': "[\'MID_LANE\']"}, {'blueFirstTowerLane': 2})
    result = result.replace({'blueFirstTowerLane': "[\'BOT_LANE\']"}, {'blueFirstTowerLane': 3})
    result = result.replace({'blueFirstTowerLane': "[]"}, {'blueFirstTowerLane': 0})
    result = result.replace({'redFirstTowerLane': "[\'TOP_LANE\']"}, {'redFirstTowerLane': 1})
    result = result.replace({'redFirstTowerLane': "[\'MID_LANE\']"}, {'redFirstTowerLane': 2})
    result = result.replace({'redFirstTowerLane': "[\'BOT_LANE\']"}, {'redFirstTowerLane': 3})
    result = result.replace({'redFirstTowerLane': "[]"}, {'redFirstTowerLane': 0})

    result['blueAirDragon'] = 0
    result['blueEarthDragon'] = 0
    result['blueWaterDragon'] = 0
    result['blueFireDragon'] = 0
    result['blueElderDragon'] = 0
    result['redAirDragon'] = 0
    result['redEarthDragon'] = 0
    result['redWaterDragon'] = 0
    result['redFireDragon'] = 0
    result['redElderDragon'] = 0


    for i in result.index:
        temp = result.loc[i, 'blueDragonType']
        splitted = temp[1:-1].split(', ')
        for dragon in splitted:
            
            if(dragon == "'AIR_DRAGON'"):
                result.loc[i, 'blueAirDragon'] = result.loc[i, 'blueAirDragon'] + 1
            elif(dragon == "'EARTH_DRAGON'"):
                result.loc[i, 'blueEarthDragon'] = result.loc[i, 'blueEarthDragon'] + 1
            elif(dragon == "'WATER_DRAGON'"):
                result.loc[i, 'blueWaterDragon'] = result.loc[i, 'blueWaterDragon'] + 1
            elif(dragon == "'FIRE_DRAGON'"):
                result.loc[i, 'blueFireDragon'] = result.loc[i, 'blueFireDragon'] + 1
            elif(dragon == "'ELDER_DRAGON'"):
                result.loc[i, 'blueElderDragon'] = result.loc[i, 'blueElderDragon'] + 1
                    
    
    for i in result.index:
        val = result.loc[i, 'redDragonType']
        splitted = val[1:-1].split(', ')
        for dragon in splitted:
            if(dragon == "'AIR_DRAGON'"):
                result.loc[i, 'redAirDragon'] = result.loc[i, 'redAirDragon'] + 1
            elif(dragon == "'EARTH_DRAGON'"):
                result.loc[i, 'redEarthDragon'] = result.loc[i, 'redEarthDragon'] + 1
            elif(dragon == "'WATER_DRAGON'"):
                result.loc[i, 'redWaterDragon'] = result.loc[i, 'redWaterDragon'] + 1
            elif(dragon == "'FIRE_DRAGON'"):
                result.loc[i, 'redFireDragon'] = result.loc[i, 'redFireDragon'] + 1
            elif(dragon == "'ELDER_DRAGON'"):
                result.loc[i, 'redElderDragon'] = result.loc[i, 'redElderDragon'] + 1



    result['blueChamps0'] = 0
    result['blueChamps1'] = 0
    result['blueChamps2'] = 0
    result['blueChamps3'] = 0
    result['blueChamps4'] = 0
    result['redChamps0'] = 0
    result['redChamps1'] = 0
    result['redChamps2'] = 0
    result['redChamps3'] = 0
    result['redChamps4'] = 0

    for i in result.index:
        val = result.loc[i, 'blueChamps']
        # splitted = val[1:-1].split(', ')
        splitted = val[1:-1].split(',')
        index = 0
        for champ in splitted:
            col_name = 'blueChamps' + str(index)
            # result.loc[i, col_name] = champ[1:-1]
            result.loc[i, col_name] = champ
            index += 1


    for i in result.index:
        val = result.loc[i, 'redChamps']
        # splitted = val[1:-1].split(', ')
        splitted = val[1:-1].split(',')
        index = 0
        for champ in splitted:
            col_name = 'redChamps' + str(index)
            # result.loc[i, col_name] = champ[1:-1]
            result.loc[i, col_name] = champ
            index += 1

    result = result.drop(['redChamps'], axis=1)
    result = result.drop(['blueDragonType'], axis=1)
    result = result.drop(['blueChamps'], axis=1)
    result = result.drop(['redDragonType'], axis=1)

    return result

if __name__ == "__main__" :
    for i in tqdm(range(1, 26)):
        result = pd.read_csv('Data/MatchData/TimeLine/match_data_' + str(i) + '.csv')
        result = processing(result)
        result.to_csv('Data/MatchData/TimeLine/match_data_' + str(i) + '.csv', index=False)


