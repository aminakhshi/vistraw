#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 11:03:47 2022

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


class CNN(nn.Module):
    def __init__(self, in_channels = 1, n_classes=2, n_channels=5, n_layers=3,
                 final_layer = 32, kernel=5, stride=1, seed=0):
        super(CNN, self).__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = n_channels
        self.n_classes = n_classes
        self.final_layer = final_layer
        self.kernel = kernel
        self.stride = stride

        # (batch, channels, length)
        layers_dict = collections.OrderedDict()
        layers_dict['norm0'] = nn.BatchNorm2d(self.in_channels)
        layers_dict["conv0"] = nn.Conv2d(in_channels=self.in_channels,
                                         out_channels=self.out_channels,
                                         kernel_size=self.kernel,
                                         stride=self.stride)
        layers_dict["relu0"] = nn.ReLU()
        layers_dict["pool0"] = nn.AvgPool2d((self.kernel//2, self.kernel//2))

        # last_size = 0
        for l in range(1, n_layers-1):
            layers_dict["conv{}".format(l)] = nn.Conv2d(in_channels=self.out_channels,
                                                        out_channels=self.out_channels,
                                                        kernel_size=self.kernel,
                                                        stride=self.stride)
            layers_dict["relu{}".format(l)] = nn.ReLU()
            layers_dict["pool{}".format(l)] = nn.AvgPool2d((self.kernel//2, self.kernel//2))
            # last_size=(self.out_channels//(l+1))
            # print(last_size)
        layers_dict['norm{n_layers}'] = nn.BatchNorm2d(self.out_channels)
        layers_dict["conv{n_layers}"] = nn.Conv2d(in_channels=self.out_channels,
                                             out_channels=self.out_channels//2,
                                             kernel_size=self.kernel,
                                             stride=self.stride)
        layers_dict["relu{n_layers}"] = nn.ReLU()
        layers_dict["pool{n_layers}"] = nn.AvgPool2d((self.kernel//2, self.kernel//2))
        layers_dict["flat{n_layers}"] = nn.Flatten()
        layers_dict["pool_f"] = nn.AdaptiveAvgPool1d(self.final_layer)
        layers_dict["dense_f"] = nn.Linear(self.final_layer, self.final_layer//4)
        layers_dict["relu_f"] = nn.ReLU()
        layers_dict["dense_out"] = nn.Linear(self.final_layer//4, self.n_classes)
        
        self.layers = nn.Sequential(layers_dict)

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

class DataWrapper(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.label[index]


class ModelClass():

    def __init__(self, model, n_epoch=5, batch_size=32, device="cpu", seed=0):
        if (device == "cuda") and (torch.cuda.device_count() > 1):
            model = nn.DataParallel(model)
        self.model = model.to(device)  
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.device = device
        self.seed = seed

    def predict(self, X):

        dataset = DataWrapper(X[:, None, :, :], np.zeros(len(X)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
        self.model.eval()
        all_pred_prob = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_x, input_y = tuple(t.to(self.device) for t in batch)
                pred = self.model(input_x)
                pred = F.softmax(pred, dim=1)
                all_pred_prob.append(pred.cpu().data.numpy())

        all_pred_prob = np.concatenate(all_pred_prob)
        # all_pred = np.argmax(all_pred_prob, axis=1)
        return all_pred_prob

    def fit(self, X, labels):
        torch.manual_seed(self.seed)
        class_weight = pd.Series(labels).value_counts().values
        class_weight = 1 / class_weight / np.max(1 / class_weight)
        print("class_weight", class_weight)
        ratio = 0.80
        total = len(X)
        dataset = DataWrapper(X[:int(len(X) * ratio), None, :, :], labels[:int(len(X) * ratio)])
        dataset_valid = DataWrapper(X[int(len(X) * ratio):, None, :, :], labels[int(len(X) * ratio):])

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 pin_memory=(self.device == "cuda"))
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=self.batch_size, drop_last=False)

        # train and test
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        # loss_func = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weight).to(self.device))
        loss_func = torch.nn.BCEWithLogitsLoss()
        best = {}
        best["best_valid_score"] = 99999
        for i in range(self.n_epoch):
            self.model.train()
            losses = []
            with tqdm(dataloader, unit="batch") as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {i}")
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    pred = self.model(input_x)
                    pred = F.softmax(pred, dim=1)
                    pred = pred.argmax(dim=1)
                    loss = loss_func(pred, input_y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.item())
                    #                 if (batch_idx % 10 == 0) and (len(losses) > 0):
                    #                     print("-",np.mean(losses))
                    losses.append(loss.detach().item())

            scheduler.step(i)

            # test
            self.model.eval()
            all_pred_prob = []
            all_pred_gt = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader_valid):
                    input_x, input_y = tuple(t.to(self.device) for t in batch)
                    pred = self.model(input_x)
                    all_pred_prob.append(pred.cpu().data.numpy())
                    all_pred_gt.append(input_y.cpu().data.numpy())

            all_pred_prob = np.concatenate(all_pred_prob)
            all_pred_gt = np.concatenate(all_pred_gt)
            all_pred = np.argmax(all_pred_prob, axis=1)

            bacc = sklearn.metrics.balanced_accuracy_score(all_pred_gt, all_pred)

            print("loss", np.mean(losses), "valid_bacc", bacc)

            if (best["best_valid_score"] > bacc):
                best["best_model"] = self.model.state_dict()
                best["best_valid_score"] = bacc

        self.model.load_state_dict(best["best_model"])  