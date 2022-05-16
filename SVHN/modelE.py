import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ModelE(nn.Module):
    def __init__(self):
        super(ModelE, self).__init__()
        
        self.layer_conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.layer_conv2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.layer_conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.layer_conv4 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.layer_conv5 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.layer_conv6 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1))

        self.layer_conv9 = nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.layer_conv10 = nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=(7,7))

        self.layer_fc1 = nn.Linear(in_features=25088, out_features=4096, bias=True)
        self.layer_fc2 = nn.Linear(in_features=4096, out_features=4096, bias=True)
        self.layer_fc3 = nn.Linear(in_features=4096, out_features=10, bias=True)

        self.layer_pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(0,0),dilation=(1,1),ceil_mode=False)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
    
    def forward(self, x):
        x = F.relu(self.layer_conv1(x))
        x = F.relu(self.layer_conv2(x))
        x = self.layer_pool(x)
        x = F.relu(self.layer_conv3(x))
        x = F.relu(self.layer_conv4(x))
        x = self.layer_pool(x)
        x = F.relu(self.layer_conv5(x))
        x = F.relu(self.layer_conv6(x))
        x = F.relu(self.layer_conv6(x))
        x = F.relu(self.layer_conv6(x))
        x = self.layer_pool(x)
        x = F.relu(self.layer_conv9(x))
        x = F.relu(self.layer_conv10(x))
        x = F.relu(self.layer_conv10(x))
        x = F.relu(self.layer_conv10(x))
        x = self.layer_pool(x)
        x = F.relu(self.layer_conv10(x))
        x = F.relu(self.layer_conv10(x))
        x = F.relu(self.layer_conv10(x))
        x = F.relu(self.layer_conv10(x))
        x = self.layer_pool(x)
        x = self.avg_pooling(x)
        x = F.relu(self.layer_fc1(x))
        x = self.dropout(x)
        x = F.relu(self.layer_fc2(x))
        x = self.dropout(x)
        x = F.softmax(x)
        return x