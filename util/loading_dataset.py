#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loading_dataset.py
Created on Thu May  3 12:47:36 2018

@author: sungkyun
"""
import torch
from torch.utils.data.dataset import Dataset
#from torch import from_numpy
import numpy as np
import pandas as pd
#from sklearn import preprocessing
#from sklearn.preprocessing import StandardScaler
#from sklearn.externals import joblib
import glob

from nnmnkwii import minmax_scale, scale 



DIM_INDEX = dict()
DIM_INDEX['linguistic'] = np.arange(0,420) # source: /linguistic
DIM_INDEX['f0'] = [0]                        # source: /pyworld
DIM_INDEX['log-f0'] = [1]                    # source: /pyworld
DIM_INDEX['vuv'] = [2]                       # source: /pyworld
DIM_INDEX['bap'] = [3]                       # source: /pyworld
DIM_INDEX['melcep'] = np.arange(4,64)      # source: /pyworld
DIM_INDEX['pyspec'] = np.arange(64,577)    # source: /pyworld
DIM_INDEX['melspec'] = np.arange(0, 128)   # source: /melmfcc
DIM_INDEX['mfcc'] = np.arange(128,153)     # source: /melmfcc

class CmuArcticDataset(Dataset):
    def __init__(self, data_root_dir=None, random_zpad=bool, cond_feature_select=None, transform=None):
        #data_root_dir = 'data/processed_slt_arctic/TRAIN/'
        #data_root_dir = 'data/processed_slt_arctic/TEST/'
        
        self.mulaw_filepaths = sorted(glob.glob(data_root_dir + 'mulaw/*.npy'))
        self.linguistic_filepaths = sorted(glob.glob(data_root_dir + 'linguistic/*.npy'))
        self.melmfcc_filepaths = sorted(glob.glob(data_root_dir + 'melmfcc/*.npy'))
        self.pyworld_filepaths = sorted(glob.glob(data_root_dir + 'pyworld/*.npy'))
        self.file_ids = [path.split('/')[-1][:-4] for path in self.mulaw_filepaths]
        self.random_zpad = random_zpad
        self.cond_feature_select = cond_feature_select # ['linguistic', 'f0', 'log-f0', 'vuv','bap', 'melcep', 'pyspec', 'melspec', 'mfcc']
        self.transform = transform
        
        self.scale_factor = np.load(data_root_dir + '../scale_factors.npy')
        
        # Construct conditional feature selection info
        global DIM_INDEX
        self.cond_info = dict()
        self.cond_dim = 0 # total dimension of condition features
        for sel in self.cond_feature_select:
            self.cond_info[sel] = np.arange(self.cond_dim, self.cond_dim + len(DIM_INDEX[sel]))
            self.cond_dim += len(DIM_INDEX[sel])
        
    def __getitem__(self, index):
        # Get 3 items: (file_id, mulaw, cond)
        file_id = self.file_ids[index]
        
        x = np.load(self.mulaw_filepaths[index])                 # size(x) = (T,)
        cond = np.empty((len(x),0), np.float16)                 # size(cond) = (T,d)
        
        
        cond_linguistic, cond_pyworld, cond_melmfcc = [], [], []
        if any(sel in self.cond_feature_select for sel in ['linguistic']):
            cond_linguistic = np.load(self.linguistic_filepaths[index])
        if any(sel in self.cond_feature_select for sel in ['f0', 'log-f0', 'vuv', 'bap', 'melcep', 'pyspec']):
            cond_pyworld = np.load(self.pyworld_filepaths[index])    
        if any(sel in self.cond_feature_select for sel in ['melspec', 'mfcc']):
            cond_melmfcc = np.load(self.melmfcc_filepaths[index])
        
        global DIM_INDEX
        for sel in self.cond_feature_select:
            if sel is 'linguistic':
                cond = np.hstack((cond, cond_linguistic))
            elif sel in ['f0', 'log-f0', 'vuv', 'bap', 'melcep', 'pyspec']:
                cond = np.hstack((cond, cond_pyworld[:,DIM_INDEX[sel]]))
            elif sel in ['melspec', 'mfcc']:
                cond = np.hstack((cond, cond_melmfcc[:,DIM_INDEX[sel]]))
        
        assert(cond.shape[1]==self.cond_dim) # check if stacked cond feature size mismatches
        
        
        # Feature-scaling
        cond = self.featScaler(cond)
        
        # Transpose
        cond = np.transpose(cond)       # size(cond) = (T,d) --> (d, T): required for pytorch  dataloading
        
        # Random zeropadding 20~50%
        if self.random_zpad is True:
            zpad_sz = int(len(x) * np.random.uniform(0.2,0.5)) 
            x[0:zpad_sz] = 128  # fill first <zpad_sz> samples with zeros (in mulaw-enc, 128)
            cond[:,0:zpad_sz] = 0.
        return file_id, torch.LongTensor(x), cond
    
    
    
    def featScaler(self, feat):
        
        for sel in self.cond_feature_select:
            if sel is 'linguistic':
                feat[:,self.cond_info[sel]] = minmax_scale(feat[:,self.cond_info[sel]],
                    self.scale_factor['linguistic_min'], self.scale_factor['linguistic_max'], feature_range=(0.01, 0.99))

        return feat
    
    
    def __len__(self):
        return len(self.file_ids) # return the number of examples that we have
    
    
    
class YesNoDataset(Dataset):
    def __init__(self, csv_path=None, zpad_target_len=int, transform=None):
        # Internal variables
        #csv_path = 'data/processed_yesno/test.csv'
        #csv_path = 'data/processed_yesno/train.csv'
        self.zpad_target_len = zpad_target_len
        self.transform = transform
        self.file_ids = None
        self.mulaw_filepaths = None
        self.mfcc_filepaths = None
        
        # Reading .csv file
        df = pd.read_csv(csv_path, index_col=0) # ['file_id', 'mulaw_filepath', 'mfcc_filepath']
        self.file_ids = df.iloc[:,0] 
        self.mulaw_filepaths = df.iloc[:,1]
        self.mfcc_filepaths = df.iloc[:,2]
        
        
    def __getitem__(self, index):
        # Get 3 items: (file_id, x = mulaw, cond = mfcc)
        file_id = self.file_ids[index]
        x = np.load(self.mulaw_filepaths[index]) # size = (T,)
        cond = np.load(self.mfcc_filepaths[index]) # size = (25,T)
        
        if self.zpad_target_len:
            x_length = x.shape[0]
            if x_length > self.zpad_target_len:
                x = x[0:self.zpad_target_len]
            elif x_length < self.zpad_target_len:
                zpad_sz = self.zpad_target_len - x_length
                x = np.pad(x, (zpad_sz,0), mode='constant', constant_values=128)  # padding first 48,000 samples with zeros
             
            cond_length = cond.shape[1]
            if cond_length > self.zpad_target_len:
                cond = cond[:, 0:self.zpad_target_len]
            elif cond_length < self.zpad_target_len:
                zpad_sz = self.zpad_target_len - cond_length
                cond = np.pad(cond, ((0,0),(zpad_sz, 0)), mode='constant')
        return file_id, torch.LongTensor(x), cond
        
        
    def __len__(self):
        return len(self.file_ids) # return the number of examples that we have
    



    