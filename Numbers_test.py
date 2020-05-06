import torchvision
import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt  # для отрисовки картиночек
# %matplotlib inline

transform = transforms.Compose(
    [transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
# i=0
# for data in testloader:
#     images, labels = data
#     print(images.shape)
#     i+=1
#     # %matplotlib inline
#     print(images[0].permute(1, 2, 0))
#     plt.imshow(images[0].permute(1, 2, 0))
#     plt.show()
#     if i > 20:
#         break
classes = tuple(str(i) for i in range(10))

import torch.nn as nn
import torch.nn.functional as F  # Functional

# ЗАМЕТЬТЕ: КЛАСС НАСЛЕДУЕТСЯ ОТ nn.Module
class SimpleConvNet(nn.Module):
    def __init__(self):
        # вызов конструктора предка
        super(SimpleConvNet, self).__init__()
        # необходмо заранее знать, сколько каналов у картинки (сейчас = 1),
        # которую будем подавать в сеть, больше ничего
        # про входящие картинки знать не нужно
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=66, kernel_size=7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=66, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels=66, out_channels=132, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(num_features=132, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(in_channels=132, out_channels=264, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(num_features=264, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(264*3*3, 399)  # !!!
        self.fc2 = nn.Linear(399, 231)
        self.fc3 = nn.Linear(231, 10)
        self.bnf1 = nn.BatchNorm1d(num_features=399, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)
        self.bnf2 = nn.BatchNorm1d(num_features=231, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.pool(F.dropout2d(F.relu(self.bn1(self.conv1(x))), p=0.15, training=True))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))), p=0.15, training=True)
        x = F.dropout2d(F.relu(self.bn3(self.conv3(x))), p=0.15, training=True)
        # print(x.shape)
        x = x.view(-1, 264*3*3)  # !!!
        x = F.relu(self.bnf1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.bnf2(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = self.fc3(x)
        return F.log_softmax(x)

from tqdm import notebook

# объявляем сеть
net = SimpleConvNet()

# выбираем функцию потерь
loss_fn = torch.nn.CrossEntropyLoss()

# выбираем алгоритм оптимизации и learning_rate
learning_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
losses = []

# итерируемся
for epoch in notebook.tqdm(range(5)):
    net.train()
    running_loss = 0.0
    for i, batch in enumerate(notebook.tqdm(trainloader)):
        # так получаем текущий батч
        X_batch, y_batch = batch

        # обнуляем веса
        optimizer.zero_grad()

        # forward + backward + optimize
        y_pred = net(X_batch)
        # print(y_batch, "\n\n", y_pred)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        # выведем текущий loss
        running_loss += loss.item()
        # выведем качество каждые 2000 батчей
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            losses.append(running_loss)
            running_loss = 0.0

    # plt.figure(figsize=(10, 7))
    # plt.plot(np.arange(len(losses)), losses)
    # plt.show()

print('Обучение закончено')

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        y_pred = net(images)
        _, predicted = torch.max(y_pred, 1)

        c = (predicted == labels)
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

torch.save(net.state_dict(), "/Users/sevakabrits/PycharmProjects/cnn/s1")