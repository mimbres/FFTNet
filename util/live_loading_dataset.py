#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 05:10:03 2018

@author: sungkyun
"""
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import glob
import librosa
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
from nnmnkwii.datasets import FileDataSource, FileSourceDataset, MemoryCacheFramewiseDataset
import pyworld
import pysptk # speech signal processing toolkit
from util.utils import mu_law_encode
#%%
DATA_ROOT = 'data/slt_arctic_full_data/'

class WavSource(FileDataSource):
    # data_root: Root directory of dataset, that must contains subdirectories for *.wav
    def __init__(self, data_root=None, file_sel_range=None, target_sr = 16000, max_num_files=-1):
        self.input_wav_files = sorted(glob.glob(data_root + "/wav/*.wav")) # All *.wav files into list
        self.target_sr = target_sr
        if max_num_files is not -1 :self.input_wav_files = self.input_wav_files[:max_num_files]
        if file_sel_range is not None: self.input_wav_files = self.input_wav_files[file_sel_range[0]:file_sel_range[1]]
    
    def collect_files(self):
        # This class method is required..
        return self.input_wav_files
    
    def collect_features(self, path):
        # 1.Load audio --> 2. pre-emphasis --> 3. 8bit mu-law
        x, fs = librosa.load(path, sr=self.target_sr, mono=True, dtype=np.float16)
        return x * 1.3





class CmuArcticLiveDataset(Dataset):
    def __init__(self, data_root_dir=DATA_ROOT, train_flag=True, cond_sel='mfcc', cache_size=1000, transform=None):
        
        self.train_flag = train_flag
        self.cond_sel = cond_sel # 'mfcc' or 'pyspec'
        self.cache_size= cache_size
        self.data_root_dir = data_root_dir
        
        if self.train_flag is True:
            self.X = FileSourceDataset(WavSource(data_root=data_root_dir, file_sel_range=[0,1000]))
        else:
            self.X = FileSourceDataset(WavSource(data_root=data_root_dir, file_sel_range=[1000,1132]))
            self.cache_size = 1
            
        self.utt_lengths = [len(utt) for utt in self.X]
        self.X_raw = MemoryCacheFramewiseDataset(self.X, self.utt_lengths, self.cache_size)
        self.utt_total_length = len(self.X_raw)
        
        self.sample_start, self.sample_end = list(), list()
        
        # # This initializes self.sample_start and self.sample_end
        if self.train_flag is True:
            self.rand_flush() 
        else:
            self.init_for_test()
        
        # Feature scaling factors
        scf = np.load(self.data_root_dir + '../processed_slt_arctic/scale_factors.npy').item()
        self.pyspec_max = np.max(scf['pyworld_max'][64:64+513]) #11.159795
        self.mfcc_mean = scf['melmfcc_mean'][128:128+25]
        self.mfcc_std  = scf['melmfcc_std'][128:128+25]
        return None

    
    def rand_flush(self):
        print ('Flush: Randomize sample selection in dataset...')
        self.sample_start = [0]
        while self.sample_start[-1] < self.utt_total_length-5000:
            self.sample_start.append(self.sample_start[-1] + np.random.randint(low=2500, high=5000))

        self.sample_end = self.sample_start[1:]
        self.sample_end.append(self.utt_total_length)
        return None
    
    def init_for_test(self):
        self.sample_start = [0]
        for utt_length in self.utt_lengths:
            self.sample_start.append(self.sample_start[-1] + utt_length)
            
        self.sample_end = self.sample_start[1:]
        self.sample_end.append(self.utt_total_length)        
        assert(len(self.sample_end) == len(self.sample_start))
    
    
    
    def __getitem__(self, index):
        
        if self.train_flag is True:
            # x: Zero-padded raw_audio
            x = np.zeros(5000, dtype=np.float64)
            zpad_end = 5000 - (self.sample_end[index] - self.sample_start[index])
            x[zpad_end:] = self.X_raw[self.sample_start[index]:self.sample_end[index]].astype(np.float64)
    #        assert(len(x)==5000)
        else:
            x = self.X_raw[self.sample_start[index]:self.sample_end[index]].astype(np.float64)
    
        # x_mulaw: 8bit Mulaw encoded 
        x_mulaw = mu_law_encode(x).astype(np.uint8)
        
        # cond: features to be used as a conditional input. mfcc or pyspec
        f0, timeaxis = pyworld.dio(x, 16000, frame_period=5) #(T,)
        f0 = pyworld.stonemask(x, f0, timeaxis, 16000)
        lf0 = f0.copy()
        lf0[np.nonzero(f0)] = np.log(f0[np.nonzero(f0)]) #(T,)
        #vuv = (lf0 != 0).astype(np.uint8) #(T,)
        
        if self.cond_sel is 'pyspec':
            cond = pyworld.cheaptrick(x, f0, timeaxis, 16000) # (T,d=513)
            cond = cond / self.pyspec_max
        elif self.cond_sel is 'mfcc':
            melspec = librosa.feature.melspectrogram(y=x, sr=16000, power=2.0, n_fft=400, hop_length=80, n_mels=128)
            cond = librosa.feature.mfcc(S=librosa.power_to_db(melspec), sr=16000, n_mfcc=25) #(d=25, T)
            cond = np.transpose(cond)  # (T,d)
            cond = scale(cond, self.mfcc_mean, self.mfcc_std)
            
            
        # Stack cond
        cond = np.hstack((lf0[:,None], cond))
        
        # cond: transpose to (d,T) and Resample
        cond = librosa.core.resample(np.transpose(cond), 200, 16000, res_type='kaiser_fast', fix=True, scale=False)
        
        # Resize and transpose
        cond = librosa.util.fix_length(cond, len(x_mulaw), mode='edge') # (d,T)
        
        return index, torch.LongTensor(x_mulaw), cond.astype(np.float32)
        
    
    def __len__(self):
        return len(self.sample_start) # return the number of examples that we have


#%%