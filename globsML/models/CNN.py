import torch
from torch import nn
from torch.nn import functional as F

class CNN(nn.Module):
    def __init__(self):
        '''
        Convolutional neural network architecture.
        '''
        super().__init__()
        # create network layers
        # convolutions
        self.conv1 = nn.Conv2d(2, 80, kernel_size=4, stride=1)
        self.conv2 = nn.Conv2d(80, 40, 4, 1)
        self.conv3 = nn.Conv2d(40, 20, 2, 2)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout = nn.Dropout(p=0.05)
        # MLP (dense layers)
        self.fc1 = nn.Linear(180, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        '''
        Apply network to input x.
        Returns both the prediction as well as the
        activity of the last layer.
        '''
        # first convolutional layer
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        # second convolutional layer
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        # third convolutional layer
        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        # first dense layer
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # output layer
        xout = F.sigmoid(self.fc2(x))
        return xout, x
