import requests as req
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import numpy as np
import time
import errorDetect as ed
import sys
from tqdm import tqdm

def get_full_data(match_ids, api, minute):
    """
    match_ids: 매치 아이디가 담긴 리스트
    api: Riot에서 발급 받은 API
    minute: 분 단위의 값. 설정된 시간 값 이전까지의 매치 데이터를 불러오기 위함 
    """
    start_time = time.time()
    ### 데이터프레임 생성
    use_columns = ['gameId','blueWins', 'blueChamps','blueTotalGolds','blueCurrentGolds','blueTotalLevel'\
                       ,'blueAvgLevel','blueTotalMinionKills','blueTotalJungleMinionKills'
                       ,'blueFirstBlood','blueKill','blueDeath','blueAssist'\
                       ,'blueWardPlaced','blueWardKills','blueFirstTower','blueFirstInhibitor'\
                       ,'blueFirstTowerLane'\
                       ,'blueTowerKills','blueMidTowerKills','blueTopTowerKills','blueBotTowerKills'\
                       ,'blueInhibitor','blueFirstDragon','blueDragonType','blueDragon','blueRiftHeralds'\
                       ,'blueBaron', 'blueFirstBaron'
                       ,'redWins', 'redChamps','redTotalGolds','redCurrentGolds','redTotalLevel'\
                       ,'redAvgLevel','redTotalMinionKills','redTotalJungleMinionKills'
                       ,'redFirstBlood','redKill','redDeath','redAssist'\
                       ,'redWardPlaced','redWardKills','redFirstTower','redFirstInhibitor'\
                       ,'redFirstTowerLane'\
                       ,'redTowerKills','redMidTowerKills','redTopTowerKills','redBotTowerKills'\
                       ,'redInhibitor','redFirstDragon','redDragonType','redDragon','redRiftHeralds'\
                      , 'redBaron', 'redFirstBaron']

    
    
    match_count = 0
    
    for match_id in tqdm(match_ids):
        result = pd.DataFrame(columns = use_columns)
        
        try:
                ### 타임라인 데이터 크롤링해오기
            URL = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline?api_key={api}"
            res = req.get(URL)
            ed.errorDetect(res, URL)


            ### 가져올 시간 정의 (~분까지의 경기 데이터)
            timeline = minute*60*1000    # 분을 ms 단위로 변환
            
            frames = res.json()['info']['frames']
        

            time_point = len(frames)-1   
            
            if time_point < minute:
                continue
            
            URL = f"https://asia.api.riotgames.com/lol/match/v5/matches/{match_id}?api_key={api}"
            res1 = req.get(URL)

            ed.errorDetect(res1, URL)
                
            dict1 = res1.json()
            
            ## 블루팀 승패 & 챔피언 종류
            blue_win = dict1["info"]['teams'][0]["win"]
            red_win = dict1["info"]["teams"][1]["win"]
            
            blue_champions = []
            red_champions = []
            for i in range(len(dict1['metadata']["participants"])):
                user = dict1['info']["participants"][i]
                if user["teamId"] == 100: 
                    blue_champions.append(user["championId"])
                elif user["teamId"] == 200:
                    red_champions.append(user["championId"])
            
            ### participantFrames 에서 가져올 수 있는 데이터
            bluetotal_gold, bluecurrent_gold, bluetotal_level, bluetotal_minionkill, bluetotal_jungleminionkill = 0,[],0,0,0
            redtotal_gold, redcurrent_gold, redtotal_level, redtotal_minionkill, redtotal_jungleminionkill = 0,[],0,0,0

            ### events에서 가져올 수 있는 데이터
            blue_kill, red_kill = 0,0  # 누적 킬 수
            blue_firstkill, red_firstkill = 0,0  # 퍼스트블러드 (0 아니면 1)
            blue_assist, red_assist = 0,0  # 누적 어시스트 수
            red_death, blue_death = 0,0  # 누적 데스 
            blue_wardplace, red_wardplace = 0,0  # 누적 와드 설치 횟수
            blue_wardkill, red_wardkill = 0,0  # 누적 와드 킬 수
            blue_elite, red_elite = 0,0  # 누적 엘리트몬스터 처치 수
            blue_rift, red_rift = 0,0  # RIFTHERALD 처치 수 
            blue_dragon, red_dragon = 0,0  # DRAGON 처치 수
            blue_baron, red_baron = 0,0  #  BARON 처치 수
            blue_firstdragon, red_firstdragon = 0,0  # DRAGON 누가 먼저 먹었는지 (0 아니면 1)
            blue_dragontype, red_dragontype = [],[]  # 처치한 DRAGON 타입
            blue_firstbaron, red_firstbaron = 0,0  # BARON 누가 먼저 먹었는지 (0 아니면 1)
            blue_tower,red_tower = 0,0  # 부신 타워 수
            blue_firsttower, red_firsttower = 0,0  # 첫번째로 누가 타워 부셨는지 (0 아니면 1)
            blue_firsttowerlane, red_firsttowerlane = [],[]  # 첫번째 부신 타워 어떤 라인에 있는지  (먼저 타워를 부신 팀만 값이 있음)
            blue_midtower, red_midtower = 0,0  # 부신 미트 타워 수
            blue_toptower, red_toptower = 0,0  # 부신 탑 타워 수
            blue_bottower, red_bottower = 0,0  # 부신 봇 타워 수
            blue_inhibitor, red_inhibitor = 0,0  # 부신 억제기 수
            blue_firstinhibitor, red_firstinhibitor = 0,0  # 억제기 누가 먼저 부셨는지 (0 아니면 1) 
            
            
            frames = res.json()['info']['frames']
            participant = frames[time_point]['participantFrames']
            
            
            for i in range(1,len(participant)+1):
                    # 블루팀의 참가자들
                    if 1 <= participant[str(i)]['participantId'] <= 5:
                        bluetotal_gold += (participant[str(i)]['totalGold'])
                        bluetotal_level += (participant[str(i)]['level'])
                        bluetotal_minionkill += (participant[str(i)]['minionsKilled'])
                        bluetotal_jungleminionkill += (participant[str(i)]['jungleMinionsKilled'])


                    # 레드팀의 참가자들
                    else:
                        redtotal_gold += (participant[str(i)]['totalGold'])
                        redtotal_level += (participant[str(i)]['level'])
                        redtotal_minionkill += (participant[str(i)]['minionsKilled'])
                        redtotal_jungleminionkill += (participant[str(i)]['jungleMinionsKilled'])
                        
        
            for y in range(1, minute+1, 1):
                
                participant = frames[y]['participantFrames']
                for i in range(1,len(participant)+1):
                    # 블루팀의 참가자들
                    if 1 <= participant[str(i)]['participantId'] <= 5:
                        bluecurrent_gold.append(participant[str(i)]['currentGold'])

                    # 레드팀의 참가자들
                    else:
                        redcurrent_gold.append(participant[str(i)]['currentGold'])
                
                events = frames[y]['events']

                for x in range(len(events)):

                    # 와드 킬
                    if events[x]['type'] == 'WARD_KILL':
                        if 1 <= events[x]['killerId'] <= 5:
                            blue_wardkill += 1
                        else:
                            red_wardkill += 1

                    # 와드 생성
                    elif events[x]['type'] == 'WARD_PLACED':
                        if 1 <= events[x]['creatorId'] <= 5:
                            blue_wardplace += 1
                        else:
                            red_wardplace += 1

                    # 챔피언 킬
                    elif events[x]['type'] == 'CHAMPION_KILL': 
                        if 1 <= events[x]['killerId'] <= 5:
                            # 퍼스트블러드
                            if red_kill ==0 and blue_kill == 0:
                                blue_firstkill += 1
                            blue_kill += 1
                            try:
                                blue_assist += len(events[x]['assistingParticipantIds'])
                            except:
                                pass
                            red_death += 1
                        else:
                            # 퍼스트블러드
                            if red_kill ==0 and blue_kill == 0:
                                red_firstkill += 1
                            red_kill += 1
                            try:
                                red_assist += len(events[x]['assistingParticipantIds'])
                            except:
                                pass
                            
                            blue_death += 1

                    # 엘리트 몬스터 킬
                    elif events[x]['type'] == 'ELITE_MONSTER_KILL':
                        if 1 <= events[x]['killerId'] <= 5:
                            blue_elite += 1
                            # 엘리트 몬스터: DRAGON
                            if events[x]['monsterType']== 'DRAGON':
                                if red_dragon ==0 and blue_dragon == 0:
                                        blue_firstdragon += 1

                                blue_dragontype.append(events[x]['monsterSubType'])
                                blue_dragon += 1
                            # 엘리트 몬스터: RIFTHERALD
                            elif events[x]['monsterType']== 'RIFTHERALD':
                                blue_rift += 1
                            # 엘리트 몬스터: BARON_NASHOR
                            elif events[x]['monsterType']== 'BARON_NASHOR':
                                if red_baron ==0 and blue_dragon == 0:
                                        blue_firstbaron += 1
                                blue_baron += 1
                        else:
                            red_elite += 1
                            # 엘리트 몬스터: DRAGON
                            if events[x]['monsterType']== 'DRAGON':
                                if red_dragon ==0 and blue_dragon == 0:
                                        red_firstdragon += 1
                                red_dragontype.append(events[x]['monsterSubType'])
                                red_dragon += 1
                            # 엘리트 몬스터: RIFTHERALD
                            elif events[x]['monsterType']== 'RIFTHERALD':
                                red_rift += 1
                            # 엘리트 몬스터: BARON_NASHOR
                            elif events[x]['monsterType']== 'BARON_NASHOR':
                                if red_baron ==0 and blue_dragon == 0:
                                        red_firstbaron += 1
                                red_baron += 1

                    # 건물 처치
                    elif events[x]['type'] == 'BUILDING_KILL':
                        if 1 <= events[x]['killerId'] <= 5:
                            # 건물: 타워
                            if events[x]['buildingType'] == 'TOWER_BUILDING':
                                if red_tower == 0 and blue_tower ==0:
                                    blue_firsttower += 1
                                    blue_firsttowerlane.append(events[x]['laneType'])
                                blue_tower += 1
                                # 미드 타워
                                if events[x]['laneType'] == 'MID_LANE':
                                    blue_midtower += 1
                                # 탑 타워
                                elif events[x]['laneType'] == 'TOP_LANE':
                                    blue_toptower += 1
                                # 봇 타워
                                elif events[x]['laneType'] == 'BOT_LANE':
                                    blue_bottower += 1
                            # 건물: 억제기
                            elif events[x]['buildingType'] == 'INHIBITOR_BUILDING':
                                if red_inhibitor == 0 and blue_inhibitor == 0:
                                    blue_firstinhibitor += 1
                                blue_inhibitor += 1
                        else:
                            # 건물: 타워
                            if events[x]['buildingType'] == 'TOWER_BUILDING':
                                if red_tower == 0 and blue_tower ==0:
                                    red_firsttower += 1
                                    red_firsttowerlane.append(events[x]['laneType'])
                                red_tower += 1
                                # 미드 타워
                                if events[x]['laneType'] == 'MID_LANE':
                                    red_midtower += 1
                                # 탑 타워
                                elif events[x]['laneType'] == 'TOP_LANE':
                                    red_toptower += 1
                                # 봇 타워
                                elif events[x]['laneType'] == 'BOT_LANE':
                                    red_bottower += 1
                            # 건물: 억제기
                            elif events[x]['buildingType'] == 'INHIBITOR_BUILDING':
                                if red_inhibitor == 0 and blue_inhibitor ==0:
                                    red_firstinhibitor += 1
                                red_inhibitor += 1    
                    

                ### 여태까지 모은 정보들 종합하기
                data_list = [   
                             match_id, 
                                blue_win,
                                blue_champions,
                                bluetotal_gold,
                                sum(bluecurrent_gold),
                                bluetotal_level,
                                bluetotal_level/5.0,
                                bluetotal_minionkill,
                                bluetotal_jungleminionkill,
                                blue_firstkill,
                                blue_kill,
                                blue_death,
                                blue_assist,
                                blue_wardplace,
                                blue_wardkill,
                                blue_firsttower,
                                blue_firstinhibitor,
                                blue_firsttowerlane,
                                blue_tower,
                                blue_midtower,
                                blue_toptower,
                                blue_bottower,
                                blue_inhibitor,
                                blue_firstdragon,
                                blue_dragontype,
                                blue_dragon,
                                blue_rift,
                                blue_baron,
                                blue_firstbaron,
                                red_win,
                                red_champions,
                                redtotal_gold,
                                sum(redcurrent_gold),
                                redtotal_level,
                                redtotal_level/5.0,
                                redtotal_minionkill,
                                redtotal_jungleminionkill,
                                red_firstkill,
                                red_kill,
                                red_death,
                                red_assist,
                                red_wardplace,
                                red_wardkill,
                                red_firsttower,
                                red_firstinhibitor,
                                red_firsttowerlane,
                                red_tower,
                                red_midtower,
                                red_toptower,
                                red_bottower,
                                red_inhibitor,
                                red_firstdragon,
                                red_dragontype,
                                red_dragon,
                                red_rift,
                                red_baron,
                                red_firstbaron]

            ### 추가 될 데이터프레임 생성
                df_match = pd.DataFrame([data_list], columns = use_columns)
                
                
                df_match.to_csv(f'Data/MatchData/TimeLine/match_data_{y}.csv', mode='a', index=False, header=False)

                bluecurrent_gold = [] # bluecurrent_gold 초기화
                redcurrent_gold = [] # redcurrent_gold 초기화
                
            match_count+=1
            print("완료된 match 개수: ", match_count)
             
    
        # 알 수 없는 에러가 발생했을 때 그냥 무시
        except Exception as e:
            print("에러 발생 줄 번호: {}".format(sys.exc_info()[-1].tb_lineno))
            print("알 수 없는 에러: ", e)
            
    print("총 소요된 시간: {} 초".format(np.round(time.time()-start_time)))
    return 
    

