#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocessing_yesno.py
Created on Thu May  3 00:15:48 2018

This code saves:
    - apply zero padding to the first 48,000 samples
    - [mu-law encoded audio] to <out_filedir>/enc
    - [mfcc] to <out_filedir>/mfcc
    - NOT IMPLEMENTED YET ([f0] to <out_filedir>/f0  *)
    
@author: sungkyun

"""

import argparse
import numpy as np
import pandas as pd # required for generating .csv files
import librosa # required for audio pre-processing, loading mp3 (sudo apt-get install libav-tools)
import glob, os # required for obtaining test file ID
from util.utils import mu_law_encode 


#%% Argument Parser
parser = argparse.ArgumentParser(description='Audio Preprocessing for yesno dataset')
parser.add_argument('-sr', '--sr', type=int, default=16000, metavar='N',
                    help='target sampling rate, default 16000')
parser.add_argument('-zp', '--zero_pad', type=int, default=48000, metavar='N',
                    help='target sampling rate, default 48000')
parser.add_argument('-i', '--input_filedir', type=str, default='data/waves_yesno/', metavar='N',
                    help='input source dataset directory, default=data/waves_yesno/')
parser.add_argument('-o', '--out_filedir', type=str, default='data/processed_yesno/', metavar='N',
                    help='output file directory(root of .wav subdirectories and .csv file), default=data/processed_yesno/')
args = parser.parse_args()


input_file_dir = args.input_filedir
output_file_dir = args.out_filedir

 
#%% Function def.
def displayFeat(x_spec=np.ndarray):
    import matplotlib.pyplot as plt
    import librosa.display    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(x_spec, x_axis='time')
    return 0



#%% Preprocessing --> save <u-enc> <mfcc> as .npy    
input_file_paths = sorted(glob.glob(input_file_dir + '*.wav'))
file_ids = [path.split('/')[-1][:-4] for path in input_file_paths]

# Load audio -> mono -> resample -> mfcc -> save
if not os.path.exists(output_file_dir):
                os.makedirs(output_file_dir)
if not os.path.exists(output_file_dir + 'mulaw/'):
                os.makedirs(output_file_dir + 'mulaw/')
if not os.path.exists(output_file_dir + 'mfcc/'):
                os.makedirs(output_file_dir + 'mfcc/')
                

total_input_files = len(input_file_paths)
for i in range(total_input_files):
    x_raw, sr = librosa.load(input_file_paths[i], sr=args.sr, mono=True) # Normalize? 
    x_raw = np.pad(x_raw, (args.zero_pad,0), mode='constant')  # padding first 48,000 samples with zeros
    #x_spec = librosa.feature.melspectrogram(y=x_raw, sr=sr, power=2.0, n_fft = 400, hop_length=160, n_mels=128)
    x_spec = librosa.feature.melspectrogram(y=x_raw, sr=sr, power=2.0, n_fft = 400, hop_length=1, n_mels=128)
    x_mfcc = librosa.feature.mfcc(S=librosa.power_to_db(x_spec), sr=args.sr, n_mfcc=25)
    # displayFeat(x_spec); displayFeat(x_mfcc)
    if x_mfcc.shape[1] > len(x_raw):
        x_mfcc = x_mfcc[:,0:len(x_raw)]
    elif x_mfcc.shape[1] < len(x_raw):
        x_raw = x_raw[0:x_mfcc.shape[1]]
    
    x_mulaw = mu_law_encode(x_raw)
    
    # Save mulaw
    save_file_path_mulaw = output_file_dir + 'mulaw/' + file_ids[i] + '.npy'
    np.save(save_file_path_mulaw, x_mulaw.astype('uint8'))
    # Save mfcc 
    save_file_path_mfcc = output_file_dir + 'mfcc/' + file_ids[i] + '.npy'
    np.save(save_file_path_mfcc, x_mfcc)
    
print('Preprocessing: {} files completed.'.format(total_input_files))

#%% Train/test split --> generate .csv 
# Train/test split : 54 files for train, 6 files for test
test_id_sel = [5,11,22,38,43,55]
train_id_sel = list(set(range(60)).difference(set(test_id_sel)))

# Prepare pandas dataframes
df_test = pd.DataFrame(columns=('file_id', 'mulaw_filepath', 'mfcc_filepath'))
df_train = pd.DataFrame(columns=('file_id', 'mulaw_filepath', 'mfcc_filepath'))

for idx in test_id_sel:
    save_file_path_mulaw = output_file_dir + 'mulaw/' + file_ids[idx] + '.npy'
    save_file_path_mfcc = output_file_dir + 'mfcc/' + file_ids[idx] + '.npy'
    df_test.loc[len(df_test)] = [file_ids[idx], save_file_path_mulaw, save_file_path_mfcc]  # add a new row into DataFrame 
for idx in train_id_sel:
    save_file_path_mulaw = output_file_dir + 'mulaw/' + file_ids[idx] + '.npy'
    save_file_path_mfcc = output_file_dir + 'mfcc/' + file_ids[idx] + '.npy'
    df_train.loc[len(df_train)] = [file_ids[idx], save_file_path_mulaw, save_file_path_mfcc]  # add a new row into DataFrame 

# Save .csv
df_test.to_csv(output_file_dir + 'test.csv', encoding='utf-8')
df_train.to_csv(output_file_dir + 'train.csv', encoding='utf-8')
print('Preprocessing: generated test.csv and train.csv files in {}.'.format(output_file_dir))

