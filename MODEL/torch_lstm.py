import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report

torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_dim = 60 # feature 갯수
hidden_dim = 1536 # 은닉층 길이
output_dim = 1 # true/false
learning_rate = 0.0001
epoch = 50

testX = np.load('MODEL/testX.npy')
testY = np.load('MODEL/testY.npy')
trainX = np.load('MODEL/trainX.npy')
trainY = np.load('MODEL/trainY.npy')



val_ratio = 0.2
val_size = int(len(trainX) * val_ratio)
valX = trainX[-val_size:]
valY = trainY[-val_size:]
trainX = trainX[:-val_size]
trainY = trainY[:-val_size]


trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)
testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)
valX_tensor = torch.FloatTensor(valX)
valY_tensor = torch.FloatTensor(valY)

class LSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        x = torch.sigmoid(x)
        return x
    
net = LSTM(data_dim, hidden_dim, output_dim, 1).to(device)
criterion = torch.nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


best_loss = float('inf')
train_losses = []
val_losses = []
best_loss = float('inf')
best_epoch = 0

# 학습
for i in tqdm(range(epoch)):
    optimizer.zero_grad()
    outputs = net(trainX_tensor.to(device))
    loss = criterion(outputs, trainY_tensor.to(device))
    loss.backward()
    optimizer.step()

    # 학습 손실을 기록
    train_losses.append(loss.item())

    # 검증 손실을 계산하고 기록
    val_outputs = net(valX_tensor.to(device))
    val_loss = criterion(val_outputs, valY_tensor.to(device))
    val_losses.append(val_loss.item())
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_epoch = i
        torch.save(net.state_dict(), 'best_model.pt')


#train result
outputs = net(trainX_tensor.to(device))
outputs = outputs.cpu().detach().numpy()
outputs = np.where(outputs > 0.5, 1, 0)
print(classification_report(trainY, outputs))
print('best epoch : ', best_epoch)

# 과적합 방지 손실 그래프
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses, label='Training Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# 분당 승률 예측
# predic_list = []
# net.eval()
# for i in range(1,26):
#     temp = testX[5:6, 0:i, :]
#     testX_tensor = torch.FloatTensor(temp)
#     predic = net(testX_tensor.to(device)).item()
#     predic_list.append(predic)
# print(predic_list)
# plt.figure(figsize=(12, 6))
# plt.plot(range(1,26), predic_list, 'o-', label='Blue Win', color='blue')
# plt.plot(range(1,26), [1 - x for x in predic_list], 'o-', label='Red Win', color='red')
# plt.ylim(0, 1)
# plt.legend()
# plt.show()

# plt.legend()
# plt.show()