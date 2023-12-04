from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

trainX = np.load("MODEL/trainX.npy")
trainY = np.load("MODEL/trainY.npy")
testX = np.load("MODEL/testX.npy")
testY = np.load("MODEL/testY.npy")


trainX = trainX.reshape((trainX.shape[0], -1))
testX = testX.reshape((testX.shape[0], -1))
trainY = trainY.ravel()
testY = testY.ravel()

rf_model = RandomForestClassifier(n_estimators= 50, max_depth= 128, min_samples_split= 4, min_samples_leaf= 1)

rf_model.fit(trainX, trainY)

y_pred = rf_model.predict(testX)

report = classification_report(testY, y_pred)
print(f"Classification Report: {report}")

scores = cross_val_score(rf_model, trainX, trainY, cv=5)
print(f"Cross-validated scores: {scores.mean()}")

feature_importances = rf_model.feature_importances_.reshape((25, 60)) #25분할하여 원래 상태로 복구
feature_importances = feature_importances[0:5, :] # 원하는 시간대로 조절
feature_importances = feature_importances.mean(axis=0) # 평균값으로 조절

header = ['blueCurrentGolds', 'blueFirstBlood', 'blueKill', 'blueDeath', 'blueAssist', 'blueWardPlaced', 'blueWardKills', 'blueFirstTower', 'blueFirstInhibitor', 'blueFirstTowerLane', 'blueTowerKills', 'blueMidTowerKills', 'blueTopTowerKills', 'blueBotTowerKills', 'blueInhibitor', 'blueFirstDragon', 'blueDragon', 'blueRiftHeralds', 'blueBaron', 'blueFirstBaron', 'redCurrentGolds', 'redFirstBlood', 'redKill', 'redDeath', 'redAssist', 'redWardPlaced', 'redWardKills', 'redFirstTower', 'redFirstInhibitor', 'redFirstTowerLane', 'redTowerKills', 'redMidTowerKills', 'redTopTowerKills', 'redBotTowerKills', 'redInhibitor', 'redFirstDragon', 'redDragon', 'redRiftHeralds', 'redBaron', 'redFirstBaron', 'blueAirDragon', 'blueEarthDragon', 'blueWaterDragon', 'blueFireDragon', 'blueElderDragon', 'redAirDragon', 'redEarthDragon', 'redWaterDragon', 'redFireDragon', 'redElderDragon', 'blueChamps0', 'blueChamps1', 'blueChamps2', 'blueChamps3', 'blueChamps4', 'redChamps0', 'redChamps1', 'redChamps2', 'redChamps3', 'redChamps4']

plt.figure(figsize=(10, 10))
plt.plot(feature_importances, header)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show


