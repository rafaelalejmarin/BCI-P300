# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 16:52:24 2019

@author: alhad
"""

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
from netNew import Net3, Net5, Net10, Net_nonsquare

from utils import acc
import time
import random
from truncate_data import truncate
import math

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
        self.num_n = self.num_n - 89
        #self.root_dir = root_dir
        #self.transform = transform

    def __len__(self):
        return self.num_n

    def __getitem__(self, idx):
        
        data = torch.tensor(self.dataset[idx,:8,:].reshape(-1,8,90), dtype=torch.float32)
        label = torch.tensor(self.label[idx], dtype=torch.float32)
        sample = [data, label]

        return sample

#%%       
# Loading data from 's1_test.csv' and 's1_train.csv' 

smaller_window = False
np.set_printoptions(threshold = 5000)
    
test_backup = np.genfromtxt('s1_test.csv', delimiter=',')
if smaller_window:
    [train_backup, EEG_desired] = truncate('s1_test.csv')
    lim = int(math.floor(np.shape(train_backup)[1] / 100.0)) * 100       # round to nearest 100

else:
    train_backup = np.genfromtxt('s1_train.csv', delimiter=',')
    lim = 60000

    
train = np.copy(train_backup)
test = np.copy(test_backup)

def window_data(data, randomize=False, test=None, brain_atlas=False):
    mat = np.zeros((lim,8,90))    # change the 60 K as needed
    if randomize:

        nums = [2, 4, 6, 5, 0, 1, 3, 7]     # keep random order consisten across trials
        new_data = data[1:9, :]
        new_data = new_data[nums, :]        
        data[1:9,:] = new_data
        test_new = test[1:9, :]
        test_new = test_new[nums,:]
        test[1:9, :] = test_new
    else:
        nums = None
        test = None
    for i in range(lim):
        mat[i,:,:] = data[1:9, i:i+90]
    return mat,nums,test


labelset = train[10,89:89+lim]
randomize = False
brain_atlas = False
if randomize:
    dataset, nums, test_new = window_data(train,randomize,test,brain_atlas)
    dataset = np.float32(dataset)
    test_new = np.float32(test_new)

else:
    dataset, nums, _ = window_data(train,randomize,test,brain_atlas)
    dataset = np.float32(dataset)
    test_new = test    
#%%
batch_size = 1000       # for randoming N at a time, can change if desired

my_p300 = B300Dataset(dataset, labelset)
trainloader = DataLoader(my_p300, batch_size=batch_size, shuffle=True)

#%% Training use mini batches
#my_p300 = B300Dataset(dataset, labelset)

def Yes_No(digit,th):
    if digit > th: # Setting the threshold
        return 1
    else:
        return 0

def test_model(testSet, model,th):
    _, n = testSet.size()
    n = n - 89
    pred = torch.zeros(n)
    for i in range(n):
        curr_win = testSet[1:9, i:i+90]
        curr_win = curr_win.reshape(1,1,8,90)
        pred[i] = Yes_No(model(curr_win),th)
        #print(i)
    return pred

name = 'experiment4_new.txt'
thrs = .01
with open(name,'w+') as f:
    if nums:
        f.write(str(nums) + '\n')
    f.write('precision, usefulness, sensitivity, specificity, truePositive, falsePositive, trueNegative, falseNegative, actualPositives')
    start = time.time()     
        
    for num_iter in range(3):
        start = time.time();
        if num_iter == 0:
            net = Net3()       
            netnum = 3
        elif num_iter == 1:
            net = Net5()       
            netnum = 5
        else:
            net = Net10()       
            netnum = 10
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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
        test_result = test_model(torch.tensor(test_new, dtype=torch.float32), net, thrs)
        try:
            result = acc(test_result.numpy(), test_new[10,:], 10)
            specificity = result[3]
            precision = result[1]
            TP = result[4]
        except(ZeroDivisionError):
            specificity = 0
            precision= 0
            TP = 0
            pass
        count = 0
        while (not (0.45 <= specificity <= 0.55) and count < 10000):
            if specificity > 0.55:
                thrs += 0.01
            if specificity < 0.45:
                thrs -= 0.01
            test_result = test_model(torch.tensor(test_new, dtype=torch.float32), net, thrs)
            try:
                result = acc(test_result.numpy(), test_new[10,:], 10)
                specificity = result[3]
                precision = result[1]
                TP = result[4]
            except(ZeroDivisionError):
                pass
            count += 1
        
        exp = '\n\n standard order, threshold = ' + str(thrs) + ' column conv layer, 10 iterations, batchsize = ' + str(batch_size) + '\n\n'
        f.write(exp)
        f.write('\n\nspecificity = ' + str(specificity) + ' precision = ' + str(precision) + ' truePositives = ' + TP + '\n\n')
        end = time.time()
        print(end-start)
        f.write('\n')   
    
    ## prints out all model parameters
    for name, param in net.named_parameters():
        if param.requires_grad:
            f.write(str(name) + str(param.data))

f.close()