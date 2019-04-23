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
        label = torch.tensor(self.label[idx], dtype=torch.long)
        sample = [data, label]

        return sample


#%%       
# Loading data from 's1_test.csv' and 's1_train.csv' 

testfiles = ['s1_test.csv', 's2_test.csv', 's3_test.csv', 's4_test.csv', 's5_test.csv', 's6_test.csv', 's7_test.csv', 's8_test.csv', 's9_test.csv', 's10_test.csv']
trainfiles = ['s1_train.csv', 's2_train.csv', 's3_train.csv', 's4_train.csv', 's5_train.csv', 's6_train.csv', 's7_train.csv', 's8_train.csv', 's9_train.csv', 's10_train.csv']
orders = ['standard', 'randomized']
name = 'experiments_new_col_2.txt'
order = 0
with open(name,'w+') as f:
    f.write('precision, usefulness, sensitivity, specificity, truePositive, falsePositive, trueNegative, falseNegative, actualPositives\n\n')

    for fileIndex in range(2):
        f.write('\n\n*********************************\n\n')
        f.write('subject ' + str(fileIndex))
        smaller_window = False            
        test_backup = np.genfromtxt(testfiles[fileIndex], delimiter=',')
        if smaller_window:
            [train_backup, EEG_desired] = truncate(trainfiles[fileIndex])
            lim = int(math.floor(np.shape(train_backup)[1] / 100.0)) * 100       # round to nearest 100
        
        else:
            train_backup = np.genfromtxt(trainfiles[fileIndex], delimiter=',')
            lim = int(math.floor(np.shape(train_backup)[1] / 10000.0)) * 10000       # round to nearest 100
            
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
    
        brain_atlas = False        
        labelset = train[10,89:89+lim]
        randomize = False
            
        if randomize:
            dataset, nums, test_new = window_data(train,randomize,test,brain_atlas)
            dataset = np.float32(dataset)
            test_new = np.float32(test_new)
        else:
            dataset, nums, _ = window_data(train,randomize,test,brain_atlas)
            dataset = np.float32(dataset)
            test_new = np.float32(test)
        
        batch_size = 1000       # for randoming N at a time, can change if desired
        my_p300 = B300Dataset(dataset, labelset)
        trainloader = DataLoader(my_p300, batch_size=batch_size, shuffle=True)
    
    
    #%% Training use mini batches
    #my_p300 = B300Dataset(dataset, labelset)
    
            
        def Yes_No(out,th):
            if out[1] > th: # Setting the threshold
                return 1
            else:
                return 0
        
        def test_model(testSet, model, th):
            _, n = testSet.size()
            n = n - 89
            pred = torch.zeros(n)
            
            for i in range(n):
                curr_win = testSet[1:9, i:i+90]
                curr_win = curr_win.reshape(1,1,8,90).to(device)
                pred[i] = Yes_No(model(curr_win).detach().numpy().reshape(-1),th)
                #print(i)
            return pred
    
        th = [.01, .02]
        if nums:
            f.write(str(nums) + '\n')
        start = time.time()    
        for thrs in th:
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
            
            test_result = test_model(torch.tensor(test_new, dtype=torch.float32), net, thrs)
        
            try:
                result = acc(test_result.numpy(), test_new[10,:], 10)
                exp = '\n\n' + orders[order] + 'order, threshold = ' + str(thrs) + ' column conv layer, 10 iterations, batchsize = ' + str(batch_size) + '\n\n'
                f.write(exp)
                f.write(str(result))
            except (ZeroDivisionError):
                print('\nzero predicted positive indices')
                f.write('\n\nzero predicted positive indices for column conv layer \n')
                pass
            end = time.time()
            print(end-start)
            f.write('\n')
        
        ## prints out all model parameters  
        for name, param in net.named_parameters():
            if param.requires_grad:
                f.write(str(name) + str(param.data))
        
f.close()