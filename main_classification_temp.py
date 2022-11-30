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
import scipy.io as io
from sklearn.utils import shuffle
import sklearn, sklearn.neighbors, sklearn.model_selection
from common import modlib
import matplotlib.pylab as plt
import sklearn.model_selection as model_selection
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics

from common import utils, visualization
import datetime
from common import param_dict
import argparse

if __name__ == "__main__":
    ## TODO: Model parameters should be updated after tuning

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', default="/home/amin/khadralab/neuro/ts_raw_500", \
                        type=str, help='path file to read the single-cell dataframe')
    parser.add_argument('--output_class', default="event", type=str,
                        help='level of MultiIndex in input DataFrame to use as model output')
    parser.add_argument('--model_name', type=str, default="tsCNN1", choices= param_dict.models, help='Classifier model structure')
    parser.add_argument('--save_output', type=bool, default=True, help='Save Outputs')
    parser.add_argument('--plots', type=bool, default=True, help='return and save plots from the result')
    parser.add_argument('--results_path', default=os.path.join('/home/amin/khadralab/neuro/','vistraw', 'results'), \
                        type=str, help='path to save the result')
    parser.add_argument('--add', nargs='+', default=None, \
                        help='level of MultiIndex in input DataFrame to use as a model feature')
    parser.add_argument('--drop', nargs='+', default=None, \
                        help='Value in output_class to remove as potential output')
    parser.add_argument('--scaling', type=str, default=None, choices= param_dict.scalers, help='sklearn preprocessing scaler to use')
    parser.add_argument('--date', type=str, default=None, help='input date to read results')
    args = parser.parse_args()
    params = param_dict.params(args, cv=True, k_folds=5, n_epoch=30, batch_size=16)
    seed = params.seed
    electrodes = [i for i in range(100) if i not in [0,56,78,88,90,99]]
    X  = []
    states = [] 
    for fname in params.toward_list:
        trial = io.loadmat(fname)['temp']
        trial = trial.reshape(100,1001)
        key = fname.split('frame_')[1][:-4]
        for i in electrodes:
            X.append(trial[i,:])
            states.append((key, i, 'T'))
    for fname in params.away_list:
        trial = io.loadmat(fname)['temp']
        trial = trial.reshape(100,1001)
        key = fname.split('frame_')[1][:-4]
        for i in electrodes:
            X.append(trial[i,:])
            states.append((key, i, 'A'))
        
    ss = pd.DataFrame(states, columns=["trial_id", "mua_id", "event"])
    ss = ss.astype({"trial_id":'int', "mua_id":'int'})
    ss['event'] = pd.Categorical(ss['event'], categories=['T', 'A'], ordered=True)
    index= pd.MultiIndex.from_frame(ss, names=["trial_id", "mua_id", "event"])

    le = LabelEncoder().fit(['T', 'A'])
    y = le.transform(ss['event'])
    y = pd.Series(y, name ='True Values', index=index)     
    X = pd.DataFrame(np.array(X), index=index)
    X = X.astype('float')
    # time = np.arange(-1000,1002,2)
    X = X.iloc[:,450:551]
    X, y = shuffle(X, y, random_state=seed)
    
    y = shuffle(y)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=.75)
    kfold = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    results = {}
    for fold, (train_ids, test_ids) in enumerate(kfold.split(X=X, y=y)):
        # Start print
        print(f'Running FOLD {fold}')
        print('--------------------------------')
        params.fold = str(fold)
        X_train, X_test = X.iloc[train_ids], X.iloc[test_ids]
        y_train, y_test = y.iloc[train_ids], y.iloc[test_ids]
        # if params.scaling:
            #     train_X = utils.scale_fit(train_X, scaling=params.scaling)
            #     test_X = utils.scale_fit(test_X, scaling=params.scaling)
            #     print("train_X", train_X.shape, "test_X", test_X.shape)
            
        if params.model_name in ["knn", "lr", "adaboost"]:
            model = utils.get_model(params.model_name, mode = 'model')
            model.fit(train_X, train_y)
            fold_prediction = model.predict(test_X)
        else:
            network = modlib.tsCNN1(in_channels=1, n_classes=1, n_channels=2,
                                   kernel=3, stride=1, seed=0)
            model = modlib.ModelClass(network, device='cuda', batch_size=32,
                                      n_epoch=20, seed=0)
            
            model.fit(X_train, y_train)
            fold_prediction = model.predict(X_test)
            fold_prediction = model.predict_proba(X_test)
            test_acc = metrics.accuracy_score(y_test, np.round(fold_prediction))
            avg_per_class_acc = metrics.balanced_accuracy_score(y_test, np.round(fold_prediction))
            print('Test %d, test acc: %.6f, test avg acc: %.6f' % (fold, test_acc, avg_per_class_acc))
        if hasattr(params, 'save_output'):
            torch.save(model.model.state_dict(), os.path.join(params.model_path, params.model_name+f'best_model-fold-{fold}.pth'))
        # cat_type = CategoricalDtype(categories=y_test.index.get_level_values(level='Peptide').categories, ordered=True)

        if hasattr(params, 'save_output'):
            fold_results_name = os.path.join(params.results_path,f'predictions-fold-{fold}.pkl')
            fold_results = utils.make_results(y_test, fold_prediction, le, 'event', save_path = fold_results_name)
        else:
            fold_results = io.make_results(y_test, fold_prediction, le, params.output_class)
        
        if hasattr(params, 'plots'):
            visualization.plot_confusion_matrix(fold_results['True Values'], fold_results['Predicted Values'], 'event', params)
            visualization.plot_ovr_roc_curves(fold_results, fold_prediction, le, params.output_class, params)
            visualization.plot_prediction_distributions(fold_results, params.output_class, params)
        
        results[fold] = fold_results        




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
