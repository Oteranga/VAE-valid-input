import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

batch_size = 32

train_dataset = torchvision.datasets.SVHN(root = './dataset', split='train', transform=transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()]), download=True)

test_dataset = torchvision.datasets.SVHN(root = './dataset', split='test', transform=transforms.Compose([transforms.Resize((32, 32)),transforms.ToTensor()]), download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = batch_size,shuffle = True)


