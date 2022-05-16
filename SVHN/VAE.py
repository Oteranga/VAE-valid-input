import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layer1 = nn.Conv2d(3, 8, kernel_size=(5, 5), padding='same')#, stride=(2, 2))
        self.layer2 = nn.Conv2d(8, 16, kernel_size=(5, 5), padding='same')
        self.layer3 = nn.Conv2d(16, 32, kernel_size=(5, 5), padding='same')#, stride=(2, 2))
        self.layer4 = nn.Conv2d(32, 64, kernel_size=(5, 5), padding='same')
        self.layer5 = nn.Conv2d(64, 64, kernel_size=(5, 5), padding='same')#, stride=(2, 2))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layer1 = nn.ConvTranspose2d(64, 64, kernel_size=(5, 5), padding='same')#, stride=(2, 2))
        self.layer2 = nn.ConvTranspose2d(64, 32, kernel_size=(5, 5), padding='same')
        self.layer3 = nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), padding='same')#, stride=(2, 2))
        self.layer4 = nn.ConvTranspose2d(16, 8, kernel_size=(5, 5), padding='same')
        self.layer5 = nn.ConvTranspose2d(8, 3, kernel_size=(5, 5), padding='same')#, stride=(2, 2))

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, image):
        x = self.encoder(image)
        x = self.decoder(x)
        return x

        