import torch.nn as nn
from utils import config
from torch import flatten


class CNNForFruits(nn.Module):
    def __init__(self,number_of_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3)
        # 4*126*126
        self.bn1 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()
        self.pooling = nn.MaxPool2d(2, 2)
        # 6*63*63
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(8)
        # 8*61*61
        # 8*30.5*30.5
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)
        # 16*28*28
        # 16*14*14

        self.fc1 = nn.Linear(8*30*30, 1200)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1200, number_of_classes)

    def forward(self, x):
        x = self.pooling(self.relu(self.bn1(self.conv1(x))))
        x = self.pooling(self.relu(self.bn2(self.conv2(x))))
        # x = self.pooling(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.fc2(x)

        return x

