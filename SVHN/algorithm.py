import modelA
import modelB
import modelC
import modelE
import VAE
import torch
import torch.nn as nn
import data

lr_VGG19 = 0.1
lr_all_cnn = 0.01
epochs_all_cnn = 350
epochs_VGG19 = 100
epochs_VAE = 200
total_step = len(data.train_loader)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

modelA = modelA.ModelA().to(device)
modelB = modelB.ModelB().to(device)
modelC = modelC.ModelC().to(device)
modelE = modelE.ModelE().to(device)
modelVAE = VAE.VAE().to(device)

#for ALL-CNN models and VGG19
cost = nn.CrossEntropyLoss()
SGD_optimizer = torch.optim.SGD(modelA.parameters(), lr=lr_all_cnn, weight_decay=1e-6, momentum=0.9,nesterov=True)

#for VAE
adam_optimizer = torch.optim.Adam(modelVAE.parameters(), lr=0.01)

for epoch in range(epochs_all_cnn):
    for i, (images, labels) in enumerate(data.train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = modelA(images)
        loss = cost(outputs, labels)
        # Backward and optimize
        #SGD_optimizer.zero_grad()
        adam_optimizer.zero_grad()
        loss.backward()
        #SGD_optimizer.step()
        adam_optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, epochs_all_cnn, i+1, total_step, loss.item()))


with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in data.test_loader:
      images = images.to(device)
      labels = labels.to(device)
      outputs = modelA(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Accuracy of the network: {} %'.format(100 * correct / total))
