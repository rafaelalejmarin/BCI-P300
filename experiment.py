#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:41:44 2019

@author: ruijia
"""

import numpy as np
import torch
import pandas as pd
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from net import Net3, Net5, Net10


#%%    
class B300Dataset(Dataset):
    """B300 dataset."""

    def __init__(self, dataset, label):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset = dataset
        self.label = label
        self.num_n, self.num_ch, self.num_t = np.shape(dataset)
        self.num_n = self.num_n - 59
        #self.root_dir = root_dir
        #self.transform = transform

    def __len__(self):
        return self.num_n

    def __getitem__(self, idx):
        
        data = torch.tensor(self.dataset[idx,:8,:].reshape(-1,8,60), dtype=torch.float32)
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        sample = [data, label]

        return sample
#%%
my_p300 = B300Dataset(dataset, labelset)
trainloader = DataLoader(my_p300, batch_size=batch_size)
dataiter = iter(trainloader)
data, labels = dataiter.next()
#%%       
# Loading data from 's1_test.csv' and 's1_train.csv' 
train = np.genfromtxt('s1_train.csv', delimiter=',')
test = np.genfromtxt('s1_test.csv', delimiter=',')

# Reformat the data
def window_data(data):
    mat = np.zeros((60000,8,60))
    for i in range(60000):
        mat[i,:,:] = data[1:9, i:i+60]
    return mat

# We only take the first 60000 samples
dataset = np.float32(window_data(train))
labelset = train[10,59:60059]

batch_size = 1000

my_p300 = B300Dataset(dataset, labelset)
trainloader = DataLoader(my_p300, batch_size=batch_size, shuffle=True)

#%% Training by go through the whole dataset
net = Net()
loss = nn.MSELoss()    
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#net.to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
dataset = torch.tensor(dataset, dtype=torch.float32).to(device)
labelset = torch.tensor(labelset, dtype=torch.float32).to(device)

for iter in range(10000):
    optimizer.zero_grad()
    idx = iter * batch_size % 60000
    data = dataset[idx:idx+batch_size, :, :]
    #data = np.float32(data)
    data = data.reshape(batch_size,1,8,60)
    pred = net(data)
    label = labelset[idx:idx+batch_size]
    loss_curr = loss(label, pred)
    loss_curr.backward()
    optimizer.step()
    if iter % 10 == 0:
        print('iter = ', iter, 'loss = ', loss_curr)
        
#%% Training use mini batches
#my_p300 = B300Dataset(dataset, labelset)
net = Net10()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(labels, outputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

    
#%% Testing the result need modification if use different input
        
def Yes_No(digit):
    if digit > 0.02: # Setting the threshold
        return 1
    else:
        return 0

def test_model(testSet, model):
    _, n = testSet.size()
    n = n - 59
    pred = torch.zeros(n)
    
    for i in range(n):
        curr_win = testSet[1:9, i:i+60]
        curr_win = curr_win.reshape(1,1,8,60)
        pred[i] = Yes_No(model(curr_win))
        #print(i)
    return pred

test_result = test_model(torch.tensor(test, dtype=torch.float32), net)


#%%
from utils import acc

result = acc(test_result.numpy(), test[10,:], 100)














