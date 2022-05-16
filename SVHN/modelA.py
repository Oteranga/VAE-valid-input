import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

batch_size = 32

class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        
        self.layer1 = nn.Conv2d(3, 96, kernel_size=(5, 5), padding='same')
        self.layer2 = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2))
        self.layer3 = nn.Conv2d(96, 192, kernel_size=(5, 5), padding='same')
        self.layer4 = nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2))
        self.layer5 = nn.Conv2d(192, 192, kernel_size=(3, 3), padding='same')
        self.layer6 = nn.Conv2d(192, 192, kernel_size=(1, 1), padding='valid')
        self.layer7 = nn.Conv2d(192, 10, kernel_size=(1, 1), padding='valid')
        self.pooling = nn.AdaptiveAvgPool2d((batch_size, 10))
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = self.pooling(x)
        x = x.reshape(x.size(0), -1)
        return x


