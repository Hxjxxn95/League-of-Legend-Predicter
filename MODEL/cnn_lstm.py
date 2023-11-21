import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

torch.manual_seed(0)
torch.cuda.empty_cache()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

data_dim = 60 # feature 갯수
hidden_dim = 1536# 은닉층 길이
output_dim = 1 
learning_rate = 0.0001
iterations = 50

trainX = np.load('trainX.npy')
trainY = np.load('trainY.npy')
testX = np.load('testX.npy')
testY = np.load('testY.npy')



trainX_tensor = torch.FloatTensor(trainX)
trainY_tensor = torch.FloatTensor(trainY)
testX_tensor = torch.FloatTensor(testX)
testY_tensor = torch.FloatTensor(testY)

print(trainX_tensor.shape)
print(trainY_tensor.shape)

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Reshape input for Conv1d
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # Reshape input for LSTM
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        x = torch.sigmoid(x)
        return x

net = CNNLSTM(data_dim, hidden_dim, output_dim, 1).to(device)
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
best_loss = float('inf')

for i in tqdm(range(iterations)):
    optimizer.zero_grad()
    outputs = net(trainX_tensor.to(device))
    loss = criterion(outputs, trainY_tensor.to(device))
    loss.backward()
    optimizer.step()

net.eval()
with torch.no_grad():
    outputs = net(testX_tensor)
    predicted_labels = (outputs >= 0.5).float()

print(classification_report(testY_tensor.device(), predicted_labels.device()))