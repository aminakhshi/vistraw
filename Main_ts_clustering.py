#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 03:00:49 2022

@author: amin
"""

from sklearn.cluster import KMeans
import pickle
import logging
import os
import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn import preprocessing

from multiprocessing import Pool, Process, Manager
import scipy.io as sio
from scipy import signal
from common.utils import bpass
from common.kmc import kmc
import matplotlib.pyplot as plt

logging = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


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

def load_files(fname, shape: str = 'array', normalize: bool = False, **kwargs):
    try:      
        trial = sio.loadmat(fname)['temp']
        assert trial.ndim == 3, 'data should be in shape (nx, ny, ts)' 
        # if normalize:
        #     trial = normalization(trial, axis= 0, norm_type = 'MeanVar')
        # trial = np.nan_to_num(trial, 0)
        return trial
    except OSError as e:
        print(e)
        
def normalize(data, ntype = 'standard', axis = 0):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    assert data.ndim >= axis, "axis is bigger than number of dimensions"
    if not ntype or ntype == 'standard':
        scaler = preprocessing.StandardScaler()
    elif ntype == 'maxmin':
        scaler = preprocessing.MinMaxScaler()
        
    scaled_data = np.zeros_like(data)
    for i in range(data.shape[axis]):
        scaled_data[i, :] = scaler.fit_transform(data[i, :].reshape(-1, 1)).flatten()
    return scaled_data
        
    
    
data_type = 'toward'
data_path = os.path.join(os.getcwd(), 'ts_raw_500', data_type)
with cd(data_path):
    toward = read_files(data_path, file_type='.mat')

data_type = 'away'
data_path = os.path.join(os.getcwd(), 'ts_raw_500', data_type)
with cd(data_path):
    away = read_files(data_path, file_type='.mat')
pixels = [0, 56, 78, 88, 90, 99]    
nx, ny = 10, 10
ts = 1001    

tspan = np.linspace(-np.fix(ts/2), np.fix(ts/2),ts)
types = {'pre': np.argwhere((tspan >= -250) & (tspan < -40)).ravel(),
         'saccad': np.argwhere((tspan >= -40) & (tspan < 0)).ravel(),
         'tw': np.argwhere((tspan >= 0) & (tspan < 50)).ravel(),
         'post': np.argwhere((tspan >= 50) & (tspan < 200)).ravel()}


# idx_pre = np.argwhere((tspan > -250) & (tspan < -40)).ravel()
# idx_saccad = np.argwhere((tspan > -40) & (tspan < 0)).ravel()
# idx_tw = np.argwhere((tspan > 0) & (tspan < 50)).ravel()
# idx_post = np.argwhere((tspan >= 50) & (tspan < 200)).ravel()
def kmc_analysis(fname):    
    # ttemp_load = load_files(tfile).reshape([nx*ny,ts])
    # ttemp = np.ma.masked_equal(ttemp_load,0)
    temp = load_files(fname).reshape([nx*ny,ts])
    temp = np.delete(temp, pixels, 0)
    beta_temp = np.zeros_like(temp)
    for i in range(beta_temp.shape[0]):
        beta_temp[i,:] = bpass(temp[i,:], fs = 500, cut_low = 20, cut_high = 40, order = 0)
    # beta_temp = np.delete(beta_temp, pixels, 0)
    baselines = np.mean(beta_temp[:, types['pre']], axis = 1)
    for i in range(len(baselines)):
        beta_temp[i,:] = beta_temp[i,:] - baselines[i]
    
    df = {'pre':[], 'saccad':[], 'tw':[], 'post':[]}
    # cut_data = {}
    for tpks in types:
        coeffs, _ = kmc(df=beta_temp[:, types[tpks]].T, order=1, dt=1/500, lag=1, mode='drift')
        df[tpks] = coeffs
    return df

# kmc_ = []
# for f in away:
#     kmc_.append(kmc_analysis(f))
if __name__ == "__main__":  # confirms that the code is under main function
    # start = time.time()
    pool = Pool(16)
    away_csd = pool.map(kmc_analysis, toward)
    pool.close()
    pool.join()
#%%
df = {'pre':[], 'saccad':[], 'tw':[], 'post':[]}    
for f in away_csd:
    for key, val in f.items():
        df[key].append(val)

        # if key == 'pre':
        #     pre.append(val)
        # elif key == 'saccad':
        #     saccad.append(val)
        # elif key == 'tw':
        #     tw.append(val)
        # elif key =='post':
        #     post.append(val)

p = r"/home/amin/khadralab/neuro/vistraw_results/vistraw_drift_toward.pkl"
with open(p,'wb') as file:
    pickle.dump(df,file) 


x1 = sum(df['pre'])/len(df['pre'])
x2 = sum(df['saccad'])/len(df['saccad'])
x3 = sum(df['tw'])/len(df['tw'])
x4 = sum(df['post'])/len(df['post'])

x1, x2, x3, x4 = x1[1:,:], x2[1:,:], x3[1:,:], x4[1:,:]
max_a = np.max([x1, x2, x3])
min_a = np.min([x1, x2, x3])
xall = [x1,x2,x3,x4]
for i in range(4):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(xall[i], interpolation='bilinear', cmap='viridis',
               vmax=1e3, vmin=-1e3)


p_toward = r"/home/amin/khadralab/neuro/vistraw_results/vistraw_drift_toward.pkl"  
with open(p_toward, 'rb') as file:
    kmc_t = pickle.load(file)


sum_t = []
for key, val in kmc_t.items():
for i, j in itertools.product(range(2), range(2)):
    

    temp = sum(val)/len(val)
    temp = temp[1:,:]
    sum_t.append(temp)
    fig, ax = plt.subplots(2, 2, figsize=(6,6))
    ax.imshow(temp, interpolation='bilinear', cmap='bwr',
               vmax=5*1e2, vmin=-5*1e2)
    plt.title('toward  ' + key)
    

p_away = r"/home/amin/khadralab/neuro/vistraw_results/vistraw_drift_away.pkl"    
with open(p_away, 'rb') as file:
    kmc_a = pickle.load(file)

sum_a = []
for key, val in kmc_a.items():
    temp = sum(val)/len(val)
    temp = temp[1:,:]
    sum_a.append(temp)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(temp, interpolation='bilinear', cmap='bwr',
               vmax=2*1e3, vmin=-2*1e3)
    plt.title('away  ' + key)
    
#%%
away_t = np.zeros([nx,ny,ts])
away_t += load_files(away[50])
pixels = [0, 56, 78, 88, 90, 99]
# away_t = away_t/len(away)
away_t = away_t.reshape([nx*ny, ts])
beta_away = np.zeros_like(away_t)
for i in range(100):
    beta_away[i,:] = bpass(away_t[i,:], fs = 500, cut_low = 20, cut_high = 40, order = 0)
    
beta_away = np.delete(beta_away, pixels, 0)


        # cut_temp[tpks] = ttemp[:, types[tpks]]
        csd_temp = np.zeros([n_ch, n_ch, 2]).astype(np.float16)
        csd_temp = csd_temp*np.nan
        if tpks == 'tw':
            for chx in range(n_ch):
                for chy in range(n_ch):
                    if chx != chy:
                        # if np.sum(ttemp[chy, types[tpks]]) != 0:
                            #     if np.sum(ttemp[chy, types[tpks]]) != 0:
                        f, cxy = signal.coherence(ttemp[chx, types[tpks]], ttemp[chy, types[tpks]], fs=500, nperseg=32)
                        idx_low = np.argwhere(f<16).ravel()
                        idx_high = np.argwhere((f>=20) & (f<=40)).ravel()
                        csd_temp[chx, chy, 0] = np.nanmean(cxy[idx_low]).astype(np.float16)
                        csd_temp[chx, chy, 1] = np.nanmean(cxy[idx_high]).astype(np.float16)
        elif tpks == 'saccad':
            for chx in range(n_ch):
                for chy in range(n_ch):
                    if chx != chy:
                        # if np.sum(ttemp[chy, types[tpks]]) != 0:
                            #     if np.sum(ttemp[chy, types[tpks]]) != 0:
                        f, cxy = signal.coherence(ttemp[chx, types[tpks]], ttemp[chy, types[tpks]], fs=500, nperseg=24)
                        idx_low = np.argwhere(f<16).ravel()
                        idx_high = np.argwhere((f>=20) & (f<=40)).ravel()
                        csd_temp[chx, chy, 0] = np.nanmean(cxy[idx_low]).astype(np.float16)
                        csd_temp[chx, chy, 1] = np.nanmean(cxy[idx_high]).astype(np.float16)        
        else:
            for chx in range(n_ch):
                for chy in range(n_ch):
                    if chx != chy:
                        # if np.sum(ttemp[chy, types[tpks]]) != 0:
                            #     if np.sum(ttemp[chy, types[tpks]]) != 0:
                        f, cxy = signal.coherence(ttemp[chx, types[tpks]], ttemp[chy, types[tpks]], fs=500, nperseg=64)
                        idx_low = np.argwhere(f<16).ravel()
                        idx_high = np.argwhere((f>=20) & (f<=40)).ravel()
                        csd_temp[chx, chy, 0] = np.nanmean(cxy[idx_low]).astype(np.float16)
                        csd_temp[chx, chy, 1] = np.nanmean(cxy[idx_high]).astype(np.float16)            
        cut_temp[tpks] =  csd_temp
    return cut_temp

data = normalize(beta_away)

coeffs, _ = kmc(df=data.T, order=1, dt=1/500, lag=1, mode='drift')

df_scaled = pd.DataFrame(data)
distortions = []
K = range(1,25)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_scaled)
    distortions.append(kmeanModel.inertia_)
    
    
from tslearn.clustering import TimeSeriesKMeans
model = TimeSeriesKMeans(n_clusters=20, metric="dtw", max_iter=50)
kvals = model.fit(data)

n, m = 3, 4
yl = 0
fig, ax = plt.subplots(n, m)
for i, j in itertools.product(range(n), range(m)):
    for xx in data[kvals == yl]:
        ax[i, j].plot(xx.ravel(), "k-", alpha=.2)
    ax[i,j].plot(model.cluster_centers_[yl].ravel(), "r-")
    ax[i,j].set_xlim(0, data.shape[1])
    # plt.ylim(-4, 4)
    yl += 1

#%%
for file in away:
    away_t += load_files(file)
away_t = away_t/len(away)
away_t = away_t.reshape([nx*ny, ts])

# away_tpass = notch_filt(away_tpass, f0 = 50, fs = 500, Q = 30)
# away_tpass = bpass_filt(away_t, 10, 45, 500)

toward_t = np.zeros([nx,ny,ts])
tt = np.zeros([len(towards),ts])
c = 0
for file in towards:
    toward_t += load_files(file)
    tt[c, :] = toward_t[4, 5, :]
    c +=1
toward_t = toward_t/len(towards)
toward_t = toward_t.reshape([nx*ny, ts])
#%%
# mask[pixels] = False
# data = np.ma.mask_rows(toward_t, pixels)  # data of a all LFP contacts (n_electrods, n_times)
sf = float(500)    # sampling frequency
data = np.delete(toward_t, pixels, axis = 0)
times = np.linspace(-1, 1, ts)*1000     # time vector

wine_value = wine_df.copy().values
min_max_scaler = preprocessing.MinMaxScaler()
wine_scaled = min_max_scaler.fit_transform(wine_value)
wine_df_scaled = pd.DataFrame(wine_scaled, columns=wine_df.columns)




from tslearn.metrics import dtw
dtw_score = dtw(x, y)


import numpy
import matplotlib.pyplot as plt

from tslearn.barycenters import \
    euclidean_barycenter, \
    dtw_barycenter_averaging, \
    dtw_barycenter_averaging_subgradient, \
    softdtw_barycenter
from tslearn.datasets import CachedDatasets

# fetch the example data set
numpy.random.seed(0)
X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")
X = X_train[y_train == 2]
length_of_sequence = X.shape[1]

from tslearn.clustering import TimeSeriesKMeans
model = TimeSeriesKMeans(n_clusters=3, metric="dtw", max_iter=10)
model.fit(X)