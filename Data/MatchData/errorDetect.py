import time
import requests as req

def errorDetect(res, URL):
    if res.status_code == 200:
                pass
    elif res.status_code == 429:
        while res.status_code != 200:
            print("API 사용량 초과. 30초 대기")
            time.sleep(30)
            res = req.get(URL)

    elif res.status_code == 503:
        while res.status_code != 200:
            print("알 수 없는 오류")
            time.sleep(30)
            res = req.get(URL)

    elif res.status_code == 403:
        while res.status_code != 200:
            print("API 갱신 필요")
            break    