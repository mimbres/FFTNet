#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
loading_dataset.py
Created on Thu May  3 12:47:36 2018

@author: sungkyun
"""
import torch
from torch.utils.data.dataset import Dataset
from torch import from_numpy
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

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
    



    