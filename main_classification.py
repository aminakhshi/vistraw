#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:57:13 2022

@author: amin
"""

import argparse
import numpy as np
import matplotlib.pylab as plt
import sys
import pandas as pd
import sklearn, sklearn.neighbors
import sklearn.model_selection as model_selection
import sklearn.linear_model, sklearn.ensemble
import collections
import gzip
from tqdm import tqdm
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os 
import pickle
import models_lib
import torch
import utils


# import encoders
class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class DataWrapper(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]
        test_X = test_X.values
        

def read_files(folder_path, file_type: str = '.mat'):
    """
    This function is designed to accept a folder path and return a list of all data files within that folder
    :param folder_path: path to a folder
    :param file_type: data format that should be parsed
    :return: list of all data files with full path
    """
    files_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(file_type):
                files_list.append(os.path.join(root, file))
    return files_list

data_type = 'toward'
data_path = os.path.join(os.path.split(os.getcwd())[0], 'ts_raw_500', data_type)
with cd(data_path):
    toward = read_files(data_path, file_type='.mat')

data_type = 'away'
data_path = os.path.join(os.path.split(os.getcwd())[0], 'ts_raw_500', data_type)
with cd(data_path):
    away = read_files(data_path, file_type='.mat')
    
    
files = {}
X  = []
y = []
for fname in toward:
    trial = io.loadmat(fname)['temp']
    trial = trial.reshape(100,1001)
    key = 'tf_'+fname.split('frame_')[1][:-4]
    files[key] = trial
    X.append(trial)
    y.append('toward')
for fname in away:
    trial = io.loadmat(fname)['temp']
    trial = trial.reshape(100,1001)
    key = 'af_'+fname.split('frame_')[1][:-4]
    files[key] = trial
    X.append(trial)
    y.append('away')

def evaluate(dir_path, results_path, model_name, seed=0, scaled=True, cross_validation=None):
    X = np.array(X)
    labels = list(set(y))
    le = LabelEncoder().fit(labels)
    y = le.transform(y)

    y = pd.Series(data=y, index=files.keys(), name='True Values')
    
    print(collections.Counter(y))

    ## TODO: Adding cross validation feature to the main repository
    if cross_validation != None:
        kfold = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    train_X, test_X, train_y, test_y = model_selection.train_test_split(X, y,
                                                                        test_size=0.2,
                                                                        stratify=y, random_state=0)
    elif model_name == "conv-basic":
    network = models_lib.CNN(
        in_channels=1,
        n_classes=len(set(train_y)),
        n_channels=5,
        n_layers=3,
        final_layer = 32,
        kernel=5,
        stride=1,
        seed=seed)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(device)
    model = models_lib.ModelClass(network, device=device, batch_size=32, n_epoch=5, seed=seed)
    model.fit(train_X, train_y)
    all_pred = model.predict(test_X)
    y_pred = np.argmax(all_pred, axis=1)
    bacc = sklearn.metrics.balanced_accuracy_score(test_y, y_pred)
    print("   Run {} " + ", Balanced Accuracy Test: {}".format(bacc))
    # if scaled:
    #     scaler = StandardScaler().fit(train_X)
    #     train_X = pd.DataFrame(data=scaler.transform(train_X), index=train_X.index, columns=train_X.columns)
    #     test_X = pd.DataFrame(data=scaler.transform(test_X), index=test_X.index, columns=test_X.columns)
    # print("train_X", train_X.shape, "test_X", test_X.shape)
