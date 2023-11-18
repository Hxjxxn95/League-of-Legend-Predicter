import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import load_model
from tensorflow.python.client import device_lib 
from keras.optimizers import Adam
from keras import backend as K

trainX = np.load('trainX.npy')
trainY = np.load('trainY.npy')
testX = np.load('testX.npy')
testY = np.load('testY.npy')

epochs = 5
model = Sequential()
K.clear_session()

from keras.callbacks import EarlyStopping
from keras.layers import TimeDistributed
early_stopping = EarlyStopping(monitor='val_loss', patience=1, mode='min', restore_best_weights=True)

model.add(LSTM(units=1024, return_sequences=True, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=512, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
optimizer = Adam(learning_rate=0.0001)
model.compile(loss="binary_crossentropy", optimizer=optimizer)

model.fit(trainX, trainY, batch_size=1, epochs=epochs, validation_split=0.1, callbacks=[early_stopping])
y_hat = model.predict(testX)


a_axis = np.arange(0, 25)
y_hat = y_hat[6,:,:] # 6번째 경기의 결과값
plt.figure(figsize=(12, 6))
plt.plot(a_axis, y_hat, 'o-', label = 'Blue Win', color = 'blue')
plt.plot(a_axis,[1 - x for x in y_hat] , 'o-', label = 'Red Win', color = 'red')
plt.ylim(0, 1)
plt.legend()
plt.show()

# 블루 팀이 이긴 게임들의 평균
# a_axis = np.arange(0, 25)
# prediction_per_minute_mean = None
# for i in range(len(testY)):
#     if testY[i][0] == 1:
#         if prediction_per_minute_mean is None:
#             prediction_per_minute_mean = y_hat[i].reshape(1, 25, 1)
#         else:
#             prediction_per_minute_mean = np.vstack((prediction_per_minute_mean, y_hat[i].reshape(1, 25, 1)))
# prediction_per_minute_mean = prediction_per_minute_mean.mean(axis=0)
# plt.figure(figsize=(12, 6))
# plt.plot(a_axis, prediction_per_minute_mean, 'o-', label='Blue Win', color='blue')
# plt.plot(a_axis, [1 - x for x in prediction_per_minute_mean], 'o-', label='Red Win', color='red')
# plt.ylim(0, 1)
# plt.legend()
# plt.show()
