#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 12:28:31 2022

@author: amin
"""


from tqdm import tqdm
import torch
import torch.optim
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import numpy as np
import sys,os
import pandas as pd
import sklearn, sklearn.model_selection, sklearn.neighbors
import sklearn.linear_model, sklearn.ensemble
import collections
from sklearn import metrics
from functools import partialmethod

    
class tsCNN1(nn.Module):
    def __init__(self, in_channels = 1, n_classes=2, n_channels=1, kernel=3, stride=1, seed=0):
        super(tsCNN1, self).__init__()
        torch.manual_seed(seed)
        self.in_channels = in_channels
        self.out_channels = n_channels
        self.n_classes = n_classes
        self.kernel = kernel
        self.stride = stride

        # (batch, channels, length)
        layers_dict = collections.OrderedDict()
        # layers_dict['norm0'] = nn.BatchNorm2d(self.in_channels)
        layers_dict["conv0"] = nn.Conv1d(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=self.kernel,
                                         stride=self.stride)
        layers_dict["relu0"] = nn.ReLU()
        layers_dict["flat0"] = nn.Flatten()
        layers_dict["dense_f1"] = nn.Linear(198, 16)
        layers_dict["relu_f1"] = nn.ReLU()
        layers_dict["dense_out"] = nn.Linear(16, self.n_classes)
        self.layers = nn.Sequential(layers_dict)

    def forward(self, x):
        x = self.layers(x)
        return x

class DataWrapper(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).long()
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.label[index]

    
def reset_weights(m):
  for layer in m.children():
      if hasattr(layer, 'reset_parameters'):
          # print(f'Reset trainable parameters of layer = {layer}')
          layer.reset_parameters()


class ModelClass():

    def __init__(self, model, n_epoch, batch_size=32, device="cpu", seed=0):
        if (device == "cuda") and (torch.cuda.device_count() > 1):
            model = nn.DataParallel(model)
        self.model = model.to(device)  
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        

    def predict(self, X, batch_size=64):
        X = X.values
        dataset = DataWrapper(X[:, None, :], np.zeros(len(X)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        test_probs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                predicted = self.model(input_x)
                predicted = torch.round(torch.sigmoid(predicted))
                test_probs.append(predicted.detach().cpu().numpy())        
        test_probs = np.concatenate(test_probs)
        return test_probs


    def predict_proba(self, X, batch_size=64):
        X = X.values
        dataset = DataWrapper(X[:, None, :], np.zeros(len(X)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        self.model.eval()
        test_probs = np.array([], dtype=float).reshape(-1)
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                predicted = self.model(input_x)
                predicted = torch.sigmoid(predicted)
                test_probs = np.concatenate((test_probs, predicted.detach().cpu().numpy().reshape(-1)), axis=0)
        return test_probs
    
    
    def fit(self, X, labels):
        torch.manual_seed(self.seed)
        X = X.values
        labels = labels.to_numpy()
        ratio = 0.80
        total = len(X)
        # class_weight = pd.Series(labels).value_counts().values
        # class_weight = 1 / class_weight / np.max(1 / class_weight)
        # print("class_weight", class_weight)
        train_dataset = DataWrapper(X[:int(len(X) * ratio), None, :], labels[:int(len(X) * ratio)])
        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=True,
                                         pin_memory=(self.device == "cuda"))
        test_dataset = DataWrapper(X[int(len(X) * ratio):, None, :], labels[int(len(X) * ratio):])
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)
        self.model.apply(reset_weights)
        # optimizer = torch.optim.SGD(self.model.parameters(),lr=1e-3)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        loss_func = torch.nn.BCEWithLogitsLoss()
        best = {}
        best["best_valid_score"] = 0.0
        for epoch in range(0, self.n_epoch):
            self.model.train()
            train_loss = 0.0
            count = 0.0
            train_pred = []
            train_true = []
            losses = []
            # print('Starting training')
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    predicted = self.model(input_x)
                    input_y = input_y.float()
                    loss = loss_func(predicted, input_y.unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    pred_val = torch.round(torch.sigmoid(predicted))
                    count += self.batch_size
                    train_loss += loss.item() * self.batch_size
                    train_true.append(input_y.cpu().numpy())
                    train_pred.append(pred_val.detach().cpu().numpy())
                    tepoch.set_postfix(loss=loss.item())

            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            # print('Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch, train_loss*1.0/count,
            #                                                                       metrics.accuracy_score(train_true, train_pred),
            #                                                                       metrics.balanced_accuracy_score(train_true, train_pred)))
            scheduler.step(epoch)
            ## TODO: Getting result_path from *params to save model in each epoch       
            # print('Training process is complete. Saving trained model.')
            # Saving the model
            # if hasattr(params, 'model_path'):
                # os.makedirs(params.model_path, exist_ok=True)
                # save_path = os.path.join(params.model_path, params.model_name + f'_fold-{params.fold}_epoch-{epoch}.pth')
                # torch.save(self.model.state_dict(), save_path)
            
            # print('Starting validating')
            self.model.eval()
            test_loss = 0.0
            count = 0.0
            test_pred = []
            test_true = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    predicted = self.model(input_x)
                    input_y = input_y.float()
                    loss = loss_func(predicted, input_y.unsqueeze(1))
                    pred_val = torch.round(torch.sigmoid(predicted))
                    count += self.batch_size
                    test_loss += loss.item() * self.batch_size
                    test_true.append(input_y.cpu().numpy())
                    test_pred.append(pred_val.detach().cpu().numpy())

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            print('Valid %d, loss: %.6f, valid acc: %.6f, valid avg acc: %.6f' % (epoch, test_loss*1.0/count,
                                                                                  test_acc, avg_per_class_acc))
            if test_acc >= best["best_valid_score"]:
                best["best_valid_score"] = test_acc
                best["best_model"] = self.model.state_dict()
                # if hasattr(params, 'model_path'):
                    # torch.save(self.model.state_dict(), os.path.join(params.model_path, params.model_name +f'best_fold-{params.fold}_epoch-{epoch}.pth'))
        
        self.model.load_state_dict(best["best_model"])
        