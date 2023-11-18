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

data_dim = 70 # feature 갯수
hidden_dim = 12 # 은닉층 길이
output_dim = 1 # true/false
learning_rate = 0.001
iterations = 5

testX = np.load('RNN/testX.npy')
testY = np.load('RNN/testY.npy')
trainX = np.load('RNN/trainX.npy')
trainY = np.load('RNN/trainY.npy')

trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)
testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

print(trainX_tensor.shape)
print(trainY_tensor.shape)
print(testX_tensor.shape)
print(testY_tensor.shape)

class Net(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(Net, self).__init__()
        self.rnn = torch.nn.GRU(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x[:, -1])
        x = torch.sigmoid(x)
        return x
    
net = Net(data_dim, hidden_dim, output_dim, 1).to(device)
criterion = torch.nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate) # Adam optimizer 사용
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
        torch.save(net, 'RNN/best_model.pth')


predY = net(testX_tensor.to(device)).to('cpu').data.numpy()
for i in range(len(predY)):
    if predY[i] > 0.5:
        predY[i] = 1
    else:
        predY[i] = 0
        
testY_tensor = testY_tensor.numpy().astype(int).tolist()
result= classification_report(testY_tensor, predY, output_dict=True) 
pprint(result)
