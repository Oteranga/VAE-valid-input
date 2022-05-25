import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ModelE(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(ModelE, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv_256_512 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.conv_512 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, padding = 'same'),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.conv_256_512(out)
        out = self.block4(out)
        out = self.conv_512(out)
        out = self.block4(out)
        
        out = out.reshape(out.size(0), -1) #Flatten
        out = self.dense(out)
        return out
