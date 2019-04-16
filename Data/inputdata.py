import numpy as np
from scipy.io import loadmat
import csv
import os

root = os.getcwd()
files = sorted(os.listdir(root))
files = sorted([f for f in files if f[-9:] == 'train.csv'])

data = {}
for k,file in enumerate(files):
    data[k] = np.genfromtxt(file,delimiter=',')
print(data[0])

data = {}
for file in files:
    data[file[:-4]] = np.genfromtxt(file,delimiter=',')
data['s1train']
# load training dataset
# s1train_csv = np.genfromtxt ('s1train.csv', delimiter=",")
# s2train_csv = np.genfromtxt ('s2train.csv', delimiter=",")
# s3train_csv = np.genfromtxt ('s3train.csv', delimiter=",")
# s4train_csv = np.genfromtxt ('s4train.csv', delimiter=",")
# s5train_csv = np.genfromtxt ('s5train.csv', delimiter=",")
# s6train_csv = np.genfromtxt ('s6train.csv', delimiter=",")
# s7train_csv = np.genfromtxt ('s7train.csv', delimiter=",")
# s8train_csv = np.genfromtxt ('s8train.csv', delimiter=",")
# s9train_csv = np.genfromtxt ('s9train.csv', delimiter=",")
# s10train_csv = np.genfromtxt ('s10train.csv', delimiter=",")


# store data in array
# s1_train_data = s1train_csv[:,:]
