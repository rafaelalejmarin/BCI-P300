#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:39:29 2019

@author: ruijia
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_nonsquare(nn.Module):
    def __init__(self):
        super(Net_nonsquare, self).__init__()
        self.conv1 = nn.Conv2d(1,10,(8,1))
        self.conv2 = nn.Conv2d(10,50,(1,15), stride = 15)
        self.fc1 = nn.Linear(50*6, 100)
        self.fc2 = nn.Linear(100, 2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(1.7159*torch.tanh(2/3*x))
        
        x = x.view(-1, 50*6)
        x = self.fc1(torch.sigmoid(x))
        x = self.fc2(torch.sigmoid(x))
        x = torch.sigmoid(x)
        return x
    
#%%
        
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Net_nonsquare()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).reshape(-1)
        

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 20 == 19:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

#%% Testing the result need modification if use different input
        
def Yes_No(out):
    if out[1] > 0.019: # Setting the threshold
        return 1
    else:
        return 0

def test_model(testSet, model):
    _, n = testSet.size()
    n = n - 89
    pred = torch.zeros(n)
    
    for i in range(n):
        curr_win = testSet[1:9, i:i+90]
        curr_win = curr_win.reshape(1,1,8,90).to(device)
        pred[i] = Yes_No(model(curr_win).detach().numpy().reshape(
                -1))
        #print(i)
    return pred

test_result = test_model(torch.tensor(test, dtype=torch.float32), net)
