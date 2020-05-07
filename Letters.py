#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import sys
from time import sleep

import torchvision
import torch
from torchvision import transforms
import cv2 as cv
from PIL import Image
import torch.utils.data as utils_data
from tqdm import notebook
import string
import pprint

import numpy as np
import matplotlib.pyplot as plt  # для отрисовки картиночек
# %matplotlib inline
BATCH_SIZE = 4
a = ord('А')

LETTERS2NUMBERS = {k: v for k,v in zip([chr(i) for i in range(a,a+6)] + [chr(a-15)] + [chr(i) for i in range(a+6,a+32)], range(33))}
NUMBERS2LETTERS = {k: v for v,k in zip([chr(i) for i in range(a,a+6)] + [chr(a-15)] + [chr(i) for i in range(a+6,a+32)], range(33))}

# np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(edgeitems=sys.maxsize)

class ImagesDs(utils_data.Dataset):
    def __init__(self, path_to_folder_with_classes, transformation=None, train=None):
        self.path_to_folder = path_to_folder_with_classes
        if train is None:
            self.data = [(self.path_to_folder + "/" + letter + "/" + f, letter) for letter in os.listdir(self.path_to_folder) if len(letter) == 1 and str(letter) != 'I' for f in os.listdir(self.path_to_folder + "/" + letter)]
        elif train:
            self.data = [(self.path_to_folder + "/" + letter + "/" + f, letter) for letter in
                         os.listdir(self.path_to_folder) if len(letter) == 1 and str(letter) != 'I' for f in
                         os.listdir(self.path_to_folder + "/" + letter)[:round(0.8 * len(os.listdir(self.path_to_folder + "/" + letter))) + 4 - round(0.8 * len(os.listdir(self.path_to_folder + "/" + letter))) % 4]]
        elif not train:
            self.data = [(self.path_to_folder + "/" + letter + "/" + f, letter) for letter in
                         os.listdir(self.path_to_folder) if len(letter) == 1 and str(letter) != 'I' for f in
                         os.listdir(self.path_to_folder + "/" + letter)[round(0.8 * len(os.listdir(self.path_to_folder + "/" + letter))) + 4 - round(0.8 * len(os.listdir(self.path_to_folder + "/" + letter))) % 4:]]
        self.transform = transformation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index]
        target = LETTERS2NUMBERS[str(target)]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = cv.imread(img, cv.IMREAD_UNCHANGED)
        img = cv.resize(img, (32, 32))
        #make mask of where the transparent bits are
        trans_mask = img[:,:,3] == 0

        #replace areas of transparency with white and not transparent
        img[trans_mask] = [255, 255, 255, 255]

        #new image without alpha channel...
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


transform = transforms.Compose([transforms.ToTensor()])

trainset = ImagesDs("data/images/Cyrillic/Cyrillic", transformation=transform, train=True)
trainloader = utils_data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = ImagesDs("data/images/Cyrillic/Cyrillic", transformation=transform, train=False)
testloader = utils_data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
# i=0
# for data in trainloader:
#     i+=1
#     images, labels = data
#     # print(images.shape)
#     cv.imshow('ddd', images[0])
#     if i > 20:
#         break
# cv.waitKey(0)
classes = [chr(i) for i in range(a,a+6)] + [chr(a-15)] + [chr(i) for i in range(a+6,a+32)]

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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=66, kernel_size=9)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=66, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(in_channels=66, out_channels=132, kernel_size=7)
        self.bn2 = nn.BatchNorm2d(num_features=132, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(in_channels=132, out_channels=264, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(num_features=264, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)
        self.fc1 = nn.Linear(264*2*2, 399)  # !!!
        self.fc2 = nn.Linear(399, 231)
        self.fc3 = nn.Linear(231, 33)
        self.bnf1 = nn.BatchNorm1d(num_features=399, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)
        self.bnf2 = nn.BatchNorm1d(num_features=231, eps=1e-07, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.pool(F.dropout2d(F.relu(self.bn1(self.conv1(x))), p=0.15, training=True))
        x = F.dropout2d(F.relu(self.bn2(self.conv2(x))), p=0.15, training=True)
        x = F.dropout2d(F.relu(self.bn3(self.conv3(x))), p=0.15, training=True)
        # print(x.shape)
        x = x.view(-1, 264*2*2)  # !!!
        x = F.relu(self.bnf1(self.fc1(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = F.relu(self.bnf2(self.fc2(x)))
        x = F.dropout(x, p=0.5, training=True)
        x = self.fc3(x)
        return F.log_softmax(x)


# объявляем сеть
net = SimpleConvNet()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # Assuming that we are on a CUDA machine, this should print a CUDA device:
#
# print(device)
#
# net.to(device)

# выбираем функцию потерь
loss_fn = torch.nn.CrossEntropyLoss()

# выбираем алгоритм оптимизации и learning_rate
learning_rate = 1e-4
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
losses = []
train_counter = []
test_losses = []

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
        # print(y_pred, "\n\n", y_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        # выведем текущий loss
        running_loss += loss.item()
        # выведем качество каждые 2000 батчей
        if i % 500 == 499:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            losses.append(running_loss)
            running_loss = 0.0

    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(len(losses)), losses)
    plt.show()

print('Обучение закончено')

class_correct = list(0. for i in range(33))
class_total = list(0. for i in range(33))

net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        y_pred = net(images)
        _, predicted = torch.max(y_pred, 1)

        c = (predicted == labels)
        # print(labels)
        # print(predicted)
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            # print(label, " ", class_correct[label])
            class_total[label] += 1

for i in range(33):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

torch.save(net.state_dict(), "/Users/sevakabrits/PycharmProjects/cnn/s2")