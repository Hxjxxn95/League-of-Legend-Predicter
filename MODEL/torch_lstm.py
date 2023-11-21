import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from pprint import pprint

torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_dim = 60 # feature 갯수
hidden_dim = 1536 # 은닉층 길이
output_dim = 1 # true/false
learning_rate = 0.0001
iterations = 50

testX = np.load('MODEL/testX.npy')
testY = np.load('MODEL/testY.npy')
trainX = np.load('MODEL/trainX.npy')
trainY = np.load('MODEL/trainY.npy')



trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)
testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

print(trainX_tensor.shape)
print(trainY_tensor.shape)
print(testX_tensor.shape)
print(testY_tensor.shape)

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


for i in tqdm(range(iterations)):
    optimizer.zero_grad()
    outputs = net(trainX_tensor.to(device))
    loss = criterion(outputs, trainY_tensor.to(device))
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(i, loss.item())
        best_loss = loss.item()
        torch.save(net, 'MODEL/best_model.pth')


net.eval()
with torch.no_grad():
    outputs = net(testX_tensor)
    predicted_labels = (outputs >= 0.5).float()

print(classification_report(testY_tensor.device(), predicted_labels.device()))


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