#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:06:22 2019

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

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        
        #self.pad = torch.nn.ReflectionPad2d(1) # Reflection padding if you want to use
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Full connection layers
        self.fc1 = nn.Linear(8*90,60) # Change the first argument accordingly if the input dimension is not 8*90
        self.fc2 = nn.Linear(60, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.relu(x))
        x = self.conv3(self.relu(x))
        
        x = x.view(-1,8*90)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv4 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv5 = nn.Conv2d(1, 1, 3, padding=1)
        
        #self.pad = torch.nn.ReflectionPad2d(1) # Reflection padding if you want to use
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Full connection layers
        self.fc1 = nn.Linear(8*90,60) # Change the first argument accordingly if the input dimension is not 8*90
        self.fc2 = nn.Linear(60, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.relu(x))
        x = self.conv3(self.relu(x))
        x = self.conv4(self.relu(x))
        x = self.conv5(self.relu(x))
        
        x = x.view(-1,8*90)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class Net10(nn.Module):
    def __init__(self):
        super(Net10, self).__init__()
        # Convolution layers
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv4 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv5 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv6 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv7 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv8 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv9 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv10 = nn.Conv2d(1, 1, 3, padding=1)
        
        #self.pad = torch.nn.ReflectionPad2d(1) # Reflection padding if you want to use
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Full connection layers
        self.fc1 = nn.Linear(8*90,60) # Change the first argument accordingly if the input dimension is not 8*90
        self.fc2 = nn.Linear(60, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.relu(x))
        x = self.conv3(self.relu(x))
        x = self.conv4(self.relu(x))
        x = self.conv5(self.relu(x))
        x = self.conv6(self.relu(x))
        x = self.conv7(self.relu(x))
        x = self.conv8(self.relu(x))
        x = self.conv9(self.relu(x))
        x = self.conv10(self.relu(x))
        
        x = x.view(-1,8*90)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
    
    
    #############################
    
    
    
#    class Net3(nn.Module):
#    def __init__(self):
#        super(Net3, self).__init__()
#        # Convolution layers
#        self.conv1 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv2 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv3 = nn.Conv2d(1, 1, 5, padding=2)
#        
#        #self.pad = torch.nn.ReflectionPad2d(1) # Reflection padding if you want to use
#        self.relu = nn.ReLU()
#        self.sigmoid = nn.Sigmoid()
#        
#        # Full connection layers
#        self.fc1 = nn.Linear(8*90,60) # Change the first argument accordingly if the input dimension is not 8*90
#        self.fc2 = nn.Linear(60, 1)
#        
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.conv2(self.relu(x))
#        x = self.conv3(self.relu(x))
#        
#        x = x.view(-1,8*90)
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
#        return x
#    
#class Net5(nn.Module):
#    def __init__(self):
#        super(Net5, self).__init__()
#        # Convolution layers
#        self.conv1 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv2 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv3 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv4 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv5 = nn.Conv2d(1, 1, 5, padding=2)
#        
#        #self.pad = torch.nn.ReflectionPad2d(1) # Reflection padding if you want to use
#        self.relu = nn.ReLU()
#        self.sigmoid = nn.Sigmoid()
#        
#        # Full connection layers
#        self.fc1 = nn.Linear(8*90,60) # Change the first argument accordingly if the input dimension is not 8*90
#        self.fc2 = nn.Linear(60, 1)
#        
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.conv2(self.relu(x))
#        x = self.conv3(self.relu(x))
#        x = self.conv4(self.relu(x))
#        x = self.conv5(self.relu(x))
#        
#        x = x.view(-1,8*90)
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
#        return x
#    
#class Net10(nn.Module):
#    def __init__(self):
#        super(Net10, self).__init__()
#        # Convolution layers
#        self.conv1 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv2 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv3 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv4 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv5 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv6 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv7 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv8 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv9 = nn.Conv2d(1, 1, 5, padding=2)
#        self.conv10 = nn.Conv2d(1, 1, 5, padding=2)
#        
#        #self.pad = torch.nn.ReflectionPad2d(1) # Reflection padding if you want to use
#        self.relu = nn.ReLU()
#        self.sigmoid = nn.Sigmoid()
#        
#        # Full connection layers
#        self.fc1 = nn.Linear(8*90,60) # Change the first argument accordingly if the input dimension is not 8*90
#        self.fc2 = nn.Linear(60, 1)
#        
#    def forward(self, x):
#        x = self.conv1(x)
#        x = self.conv2(self.relu(x))
#        x = self.conv3(self.relu(x))
#        x = self.conv4(self.relu(x))
#        x = self.conv5(self.relu(x))
#        x = self.conv6(self.relu(x))
#        x = self.conv7(self.relu(x))
#        x = self.conv8(self.relu(x))
#        x = self.conv9(self.relu(x))
#        x = self.conv10(self.relu(x))
#        
#        x = x.view(-1,8*90)
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
#        return x
#    
    