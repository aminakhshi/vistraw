#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 05:54:43 2022

@author: amin
"""
import sys,os
import collections
import datetime
import torch


models=["tsCNN1", "tsCNN2", "tabMLP", "lgb", "xgb", "conv-resnet", "conv-basic"]
scalers = ['standard', 'minmax', 'maxabs', 'robust', 'norm', 'quantile', 'power']



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


class params():
    def __init__(self, args, **kwargs):
        try:

            self.dir_path_t = os.path.join(args.file_path, 'toward')
            self.dir_path_a = os.path.join(args.file_path, 'away')
            with cd(self.dir_path_t):
                self.toward_list = read_files(self.dir_path_t, file_type='.mat')
            with cd(self.dir_path_a):
                self.away_list = read_files(self.dir_path_a, file_type='.mat')
            self.file_path = args.file_path
        except:
            raise NameError("File {} does not exist".fomrat(args.file_path))
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])
        
        self.__dict__.update( kwargs )
        if not self.date:
            self.date = datetime.datetime.now().strftime('%Y%m%d')
            
        if not hasattr(self, 'seed'):
            self.seed = 0
            
        if not hasattr(self, 'device'):
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        if hasattr(self, 'cv'):
            if not hasattr(self, 'k_folds'):
                self.k_folds = 5
        else:
            self.k_folds = 2
        
        if not hasattr(self, 'scaling'):
            print('Normalization is None. For strandard normalization choose scaling = standard')
            self.scaling = None

        if not hasattr(self, 'n_epoch'):
            self.n_epoch = 20
        
        if not hasattr(self, 'batch_size'):
            self.batch_size = 32
        
        if hasattr(self, 'save_output'):
            if not hasattr(self, 'results_path'):
                self.results_path = self.file_path
            try:
                if self.scaling != None:
                    file_name = 'classification'+'_'+self.scaling
                else:
                    file_name = 'classification'+'_'+'NoScale'
                self.results_path = os.path.join(self.results_path,file_name,self.model_name+'_'+self.date)
                os.makedirs(self.results_path, exist_ok=True)
            except:
                os.makedirs(self.results_path, exist_ok=True)
            self.model_path = os.path.join(self.results_path, 'model')
            os.makedirs(self.model_path, exist_ok=True)            