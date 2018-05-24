#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 21:18:05 2018

@author: sungkyun

INPUT:
    - <DATA_ROOT>/label_phone_align/*.lab files: using hts front-end, we generate linguistic features and silence label
    - <DATA_ROOT>/wav/*.wav files: audio files to generate acoustic features
    
OUTPUT:
    - processed_<dataset_name>/linguistic/*.npy
    - processed_<dataset_name>/mulaw/*.npy               (8-bit mulaw encoding)
    - processed_<dataset_name>/mfcc/*.npy                (25ch mfcc including energy)
    - processed_<dataset_name>/aperiodicity/*.npy        (un-/voiced)
    - processed_<dataset_name>/melspec/*.npy             (80-bin)
    -                         /spec/*.npy
    -                         /mgc/*.npy                 (mel-cepstrum)
    - processed_<dataset_name>/f0/*.npy                  (linear-interpolated f0)
    - 
"""
import argparse
import numpy as np
import librosa # required for audio pre-processing, loading mp3 (sudo apt-get install libav-tools)
import glob, os # required for obtaining test file ID
#from util.utils import mu_law_encode 
from utils import mu_law_encode

from nnmnkwii.frontend import merlin as fe # This converts HTS-style full context labels to logits based on pre-defined question files.
from nnmnkwii.preprocessing.f0 import interp1d
from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
from nnmnkwii.datasets import FileDataSource, FileSourceDataset
#from nnmnkwii.util import apply_delta_windows
from nnmnkwii.io import hts

import pysptk # speech signal processing toolkit
import pyworld # generating vocoder feature
from tqdm import tqdm # progress bar
#%% Argument Parser
parser = argparse.ArgumentParser(description='Audio and Linguistic Feature Preprocessing for CMU Arctic slt dataset')
parser.add_argument('-sr', '--sr', type=int, default=16000, metavar='N',
                    help='target sampling rate, default 16000')
parser.add_argument('-zp', '--zero_pad', type=bool, default=True, metavar='N',
                    help='left-zero-padding(make all data into same size), default=True')
parser.add_argument('-i', '--input_data_root_dir', type=str, default='data/slt_arctic_full_data/', metavar='N',
                    help='input source dataset directory, default=data/slt_arctic_full_data/')
parser.add_argument('-o', '--out_data_root_dir', type=str, default='data/processed_slt_arctic/', metavar='N',
                    help='output file directory, default=data/processed_slt_arctic/')
parser.add_argument('-max', '--max_num_files', type=int, default=-1, metavar='N',
                    help='max number of files, default=-1 (all files)')
parser.add_argument('-q', '--question_file_path', type=str, default='data/slt_arctic_full_data/questions-radio_dnn_416.hed', metavar='N',
                    help='HTS question file path, default=data/processed_slt_arctic/questions-radio_dnn_416.hed')
parser.add_argument('-sp', '--split_train_test', type=list, default=[1000,132], metavar='N',
                    help='specify numbers for splitting train and test set, default=[1000,132]')
args = parser.parse_args()


# In/Out File directories:
DATA_ROOT = args.input_data_root_dir
DST_ROOT = args.out_data_root_dir
QUESTION_PATH = args.question_file_path
DATASET_NAME = 'slt_arctic'
N_TRAIN, N_TEST = args.split_train_test # default: 1000, 132

if not os.path.exists(DST_ROOT):
    os.makedirs(DST_ROOT)
if not os.path.exists(DST_ROOT + '/TRAIN/mulaw'):
    os.makedirs(DST_ROOT +'/TRAIN/mulaw')
if not os.path.exists(DST_ROOT + '/TRAIN/linguistic'):
    os.makedirs(DST_ROOT +'/TRAIN/linguistic')
if not os.path.exists(DST_ROOT + '/TRAIN/pyworld'):
    os.makedirs(DST_ROOT +'/TRAIN/pyworld')
if not os.path.exists(DST_ROOT + '/TRAIN/melmfcc'):
    os.makedirs(DST_ROOT +'/TRAIN/melmfcc')
if not os.path.exists(DST_ROOT + '/TEST/mulaw'):
    os.makedirs(DST_ROOT +'/TEST/mulaw')
if not os.path.exists(DST_ROOT + '/TEST/linguistic'):
    os.makedirs(DST_ROOT +'/TEST/linguistic')
if not os.path.exists(DST_ROOT + '/TEST/pyworld'):
    os.makedirs(DST_ROOT +'/TEST/pyworld')
if not os.path.exists(DST_ROOT + '/TEST/melmfcc'):
    os.makedirs(DST_ROOT +'/TEST/melmfcc')

#%% Class def.
    
class SilenceSampleIdxSource(FileDataSource):

    def __init__(self, data_root=None, frame_shift_in_micro_sec=625, max_num_files=-1):
        # Build *.lab file list
        self.input_lab_files = sorted(glob.glob(DATA_ROOT + "/label_phone_align/*.lab")) # All *.lab files into list
        if max_num_files is not -1 : self.input_lab_files = self.input_lab_files[:max_num_files]
        self.frame_shift_in_micro_sec = frame_shift_in_micro_sec
        return None

    def collect_files(self):
        return self.input_lab_files

    def collect_features(self, path):
        labels = hts.load(path)
        return labels.silence_frame_indices(frame_shift_in_micro_sec=self.frame_shift_in_micro_sec)



class MulawSource(FileDataSource):
    ''' 
    8bit mulaw encoded audio, fs=16000
    Args: 
        - data_root: Root directory of dataset, that must contains subdirectories for *.wav files.
    Return:
        - x: Length x Dimension, where Dimension is 1.
        
    '''
    def __init__(self, data_root=None, target_sr = 16000, max_num_files=-1):
        
        # Build *.wav file list
        self.input_wav_files = sorted(glob.glob(DATA_ROOT + "/wav/*.wav")) # All *.wav files into list
        self.target_sr = target_sr
        if max_num_files is not -1 : self.input_wav_files = self.input_wav_files[:max_num_files]
        #self.file_ids = [path.split('/')[-1][:-4] for path in input_wav_files] # File names without extensions in a list
        return None
    
    def collect_files(self):
        # This class method is required..
        return self.input_wav_files
    
    def collect_features(self, path):
        # 1.Load audio --> 2. pre-emphasis --> 3. 8bit mu-law
        x, fs = librosa.load(path, sr=self.target_sr, mono=True, dtype=np.float64)
        x = x * 1.3
        x_mulaw = mu_law_encode(x)
        return x_mulaw.astype(np.uint8)
    
    

class LinguisticSource(FileDataSource):
    ''' 
    Linguistic features are sampled for every 5 ms.
    Args: 
        - data_root: Root directory of dataset, that must contains subdirectories for *.lab and *.wav files.
        - question_path: Question.hed filepath. 
    Return:
        - features: Length x Dimension, where Dimension is 420.
        
    '''
    def __init__(self, data_root=None, question_path=None, max_num_files=-1):
        
        # Build *.lab file list
        self.input_lab_files = sorted(glob.glob(DATA_ROOT + "/label_phone_align/*.lab")) # All *.lab files into list
        if max_num_files is not -1 : self.input_lab_files = self.input_lab_files[:max_num_files]
        #self.file_ids = [path.split('/')[-1][:-4] for path in input_lab_files] # File names without extensions in a list
        
        # Build dictionary from *.hed question file
        self.binary_dict, self.continuous_dict = hts.load_question_set(question_path)
        
        return None

    def collect_files(self):
        # This class method is required..
        return self.input_lab_files

    def collect_features(self, path):
        # 1.Load labels --> 2.Load dict from question --> 3.Parse linguistic feat.
        labels = hts.load(path)
        features = fe.linguistic_features(
                labels, self.binary_dict, self.continuous_dict,
                add_frame_features=True, subphone_features='coarse_coding') # subphone_feature = None or 'coarse_coding', coarse_coded_features[:,416,417,418,419]
        
        return features.astype(np.float32)
    
        

class PyworldSource(FileDataSource):
    ''' 
    Pyworld features are sampled for every 5 ms.
    Args: 
        - data_root: Root directory of dataset, that must contains subdirectories for *.lab and *.wav files.
        - max_num_files: default = -1 (use all)
        - target_sr: default = 16000
        - win_sz : default = 1024
        - hop_sz : default = 80
    Return:
        - features: Length x Dimension, where total Dimension is 64. 
        - features[:,0] : f0
        - features[:,1] : log-f0
        - features[:,2] : voiced/unvoiced, voiced=1, unvoiced=0
        - features[:,3] : coded aperiodicity
        - features[:,4:63] : mel-cepstrum
        - features[:,64:576] : pyworld spectrogram

    '''
    def __init__(self, data_root=None, max_num_files=-1, target_sr = 16000, win_sz=1024, hop_sz=80):
        self.target_sr = target_sr
        self.win_sz = win_sz
        self.hop_sz = hop_sz
        self.hop_sz_in_ms = int(1000*(hop_sz/target_sr))

        # Build *.wav file list
        self.input_wav_files = sorted(glob.glob(DATA_ROOT + "/wav/*.wav")) # All *.lab files into list
        if max_num_files is not -1 : self.input_wav_files = self.input_wav_files[:max_num_files]
        
        # Build *.lab file list
        self.input_lab_files = sorted(glob.glob(DATA_ROOT + "/label_phone_align/*.lab")) # All *.lab files into list
        if max_num_files is not -1 : self.input_lab_files = self.input_lab_files[:max_num_files]
        
        return None
        
    def collect_files(self):
        # This class method is required..
        return self.input_wav_files
    

    def collect_features(self, wav_path):
        
        # x: Raw audio, (Sample_length, )
        x, fs = librosa.load(wav_path, sr=self.target_sr, mono=True, dtype=np.float64)
        
        
        # f0: F0, (Frame_length, ) 
        # lf0: log(f0) --> interp1d (Frame_length, )
        # vuv: voice/unvoiced (Frame_length, )
        f0, timeaxis = pyworld.dio(x, self.target_sr, frame_period=self.hop_sz_in_ms)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        lf0 = f0.copy()
        lf0[np.nonzero(f0)] = np.log(f0[np.nonzero(f0)])
        lf0 = interp1d(lf0, kind="slinear")
        vuv = (lf0 != 0).astype(np.float32)
        
        
        # spec: Spectrogram, (Frame_length x Dim), Dim = 513
        # bap: coded aperiodicity, (Frame_length, )
        # mgc: mel-cepstrum, (Frame_length x Dim), Dim = 60
        spec = pyworld.cheaptrick(x, f0, timeaxis, fs)
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)        
        bap = pyworld.code_aperiodicity(aperiodicity, fs)
        mgc = pysptk.sp2mc(spec, order=59, alpha=pysptk.util.mcepalpha(fs))
        
        
        # Stacking Features: total dimesnion = 64
        features = np.hstack((f0[:,None], lf0[:,None], vuv[:,None], bap, mgc, spec))
        return features.astype(np.float32)



class MelspecMfccSource(FileDataSource):
    ''' 
    MFCCs features are sampled for every 5 ms.
    Args: 
        - data_root: Root directory of dataset, that must contains subdirectories for *.lab and *.wav files.
        - max_num_files: default = -1 (use all)
        - remove_silence: default = True
        - target_sr: default = 16000
        - win_sz : default = 400
        - hop_sz : default = 80
    Return:
        - features: Length x Dimension, where total Dimension is 27.  
        - features[:,0:127] : mel-spectrogram
        - features[:,128:152] : MFCCs
        
    '''
    def __init__(self, data_root=None, max_num_files=-1, target_sr = 16000, win_sz=400, hop_sz=80):
        self.target_sr = target_sr
        self.win_sz = win_sz
        self.hop_sz = hop_sz
        self.hop_sz_in_ms = int(1000*(hop_sz/target_sr))

        # Build *.wav file list
        self.input_wav_files = sorted(glob.glob(DATA_ROOT + "/wav/*.wav")) # All *.lab files into list
        if max_num_files is not -1 : self.input_wav_files = self.input_wav_files[:max_num_files]
        
        # Build *.lab file list
        self.input_lab_files = sorted(glob.glob(DATA_ROOT + "/label_phone_align/*.lab")) # All *.lab files into list
        if max_num_files is not -1 : self.input_lab_files = self.input_lab_files[:max_num_files]
        
        return None
        
    def collect_files(self):
        # This class method is required..
        return self.input_wav_files
    

    def collect_features(self, wav_path):
        
        # x: Raw audio, (Sample_length, )
        x, fs = librosa.load(wav_path, sr=self.target_sr, mono=True, dtype=np.float64)
        
        # melspec: Mel-Spectrogram, (Frame_length x Dim), Dim = 128
        # mfcc: MFCCs, (Frame_length x Dim), Dim = 25
        melspec = librosa.feature.melspectrogram(y=x, sr=fs, power=2.0, n_fft = self.win_sz, hop_length=self.hop_sz, n_mels=128)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(melspec), sr=fs, n_mfcc=25)
        melspec = np.transpose(melspec)
        mfcc = np.transpose(mfcc)
        
                
        # Stacking Features: total dimesnion = 29
        features = np.hstack((melspec, mfcc))
        return features.astype(np.float32)


#%% Main: Preprocessing

# Three data sources:
Y_mulaw      = FileSourceDataset(MulawSource(data_root=DATA_ROOT))
Y_silenceIdx = FileSourceDataset(SilenceSampleIdxSource(data_root=DATA_ROOT, frame_shift_in_micro_sec=625))
X_linguistic = FileSourceDataset(LinguisticSource(data_root=DATA_ROOT, question_path=QUESTION_PATH))
X_pyworld = FileSourceDataset(PyworldSource(data_root=DATA_ROOT))
X_melmfcc = FileSourceDataset(MelspecMfccSource(data_root=DATA_ROOT))

# Calculate Scale factors:
'''
print('Calculating scale factors: This process will take longer than 10 minutes...')
#wav_len = [len(y) for y in Y]
#y_min, y_max = minmax(Y, wav_len)

scale_factors = {}
scale_factors['linguistic_len'] = [len(x) for x in X_linguistic]
scale_factors['linguistic_min'], scale_factors['linguistic_max'] = minmax(X_linguistic, scale_factors['linguistic_len'])

scale_factors['pyworld_len'] = [len(x) for x in X_pyworld]
scale_factors['pyworld_mean'], scale_factors['pyworld_var'] = meanvar(X_pyworld, scale_factors['pyworld_len'])
scale_factors['pyworld_std'] = np.sqrt(scale_factors['pyworld_var'])
scale_factors['pyworld_min'], scale_factors['pyworld_max'] = minmax(X_pyworld, scale_factors['pyworld_len'])

scale_factors['melmfcc_mean'], scale_factors['melmfcc_var'] = meanvar(X_melmfcc, scale_factors['pyworld_len'])
scale_factors['melmfcc_std'] = np.sqrt(scale_factors['melmfcc_var'])
scale_factors['melmfcc_min'], scale_factors['melmfcc_max'] = minmax(X_melmfcc, scale_factors['pyworld_len'])

np.save(DST_ROOT + 'scale_factors.npy', scale_factors)
'''
''' To load scale_factors:
    scale_factors = np.load(DST_ROOT + 'scale_factors.npy').item()  '''
scale_factors = np.load(DST_ROOT + 'scale_factors.npy').item()
# <wav>
# Resampling silence index -->  Remove silence   --> IF Trainset: sliding window(winsz=5000, overlap=2500) -->  save .npy
#
# <Features>   
# Feature scaling 1000 train set or 132 test set
# IF Trainset: resample to 16000 with interp1D --> Removing silence  --> sliding window(winsz=5000, overlap=2500) -->  save .npy
# IF Testset : resample to 16000 with interp1D --> Removing silence --> save .npy

# Preprocess Train dataset
#ni = 0 # Index of slices(1 slice =5000 samples)  
#for i in tqdm(range(0, N_TRAIN)):
#    sil_sample_idx = Y_silenceIdx[i]
#    
#    y_mulaw = Y_mulaw[i]
#    y_mulaw = y_mulaw[:sil_sample_idx.max()+1]
#    y_len   = y_mulaw.shape[0]
#        
#    x_linguistic = X_linguistic[i]
#    x_pyworld    = X_pyworld[i]    
#    x_melmfcc    = X_melmfcc[i]        
#    
#    # Feature scaling
##    x_linguistic = minmax_scale(x_linguistic, scale_factors['linguistic_min'], scale_factors['linguistic_max'], feature_range=(0.01, 0.99))
##    x_pyworld    = scale(x_pyworld, 0, scale_factors['pyworld_std'])
##    x_melmfcc    = scale(x_melmfcc, 0, scale_factors['melmfcc_std'])  
#    
#    # Resampling fs200(5ms-hop) to fs16000
#    x_linguistic = librosa.core.resample(x_linguistic.T, 200, args.sr, res_type='kaiser_fast', fix=True, scale=False).T    
#    x_pyworld    = librosa.core.resample(x_pyworld.T, 200, args.sr, res_type='kaiser_fast', fix=True, scale=False).T 
#    x_melmfcc    = librosa.core.resample(x_melmfcc.T, 200, args.sr, res_type='kaiser_fast', fix=True, scale=False).T 
#    
#    # Reduce unlabeled index
#    x_linguistic = x_linguistic[:sil_sample_idx.max()+1]
#    x_pyworld = x_linguistic[:sil_sample_idx.max()+1]
#    x_melmfcc = x_linguistic[:sil_sample_idx.max()+1]
#    
#    # Apply 0 to silence samples
#    y_mulaw[sil_sample_idx] = 128
#    
#    # Save slices (hop=2500, win=5000)
#    sample_length = len(y_mulaw)
#    total_slices = int(np.ceil(sample_length/2500))
#    
#    y_mulaw      = librosa.util.fix_length(y_mulaw, total_slices*2500, axis=0)
#    x_linguistic = librosa.util.fix_length(x_linguistic, total_slices*2500, axis=0)
#    x_pyworld = librosa.util.fix_length(x_pyworld, total_slices*2500, axis=0)
#    x_melmfcc = librosa.util.fix_length(x_melmfcc, total_slices*2500, axis=0)
#    
#    for oneslice in range(total_slices-1):
#        fname = '{0:012d}.npy'.format(ni) # 000000000000.npy, 000000000001.npy, ...
#        slice_start_idx, slice_end_idx = oneslice * 2500, oneslice *2500 + 5000
#        
#        fpath = DST_ROOT + '/TRAIN/mulaw/' + fname # duplicating '/' is ok.
#        np.save(fpath, y_mulaw[slice_start_idx:slice_end_idx])
#        fpath = DST_ROOT + '/TRAIN/linguistic/' + fname 
#        np.save(fpath, x_linguistic.astype(np.float16)[slice_start_idx:slice_end_idx,:])
#        fpath = DST_ROOT + '/TRAIN/pyworld/' + fname 
#        np.save(fpath, x_pyworld.astype(np.float16)[slice_start_idx:slice_end_idx,:])
#        fpath = DST_ROOT + '/TRAIN/melmfcc/' + fname 
#        np.save(fpath, x_melmfcc.astype(np.float16)[slice_start_idx:slice_end_idx,:])
#        
#        ni += 1    
##    # Remove silence
##        features = np.delete(features, labels.silence_frame_indices(), axis=0)
        

# Preprocess Test dataset
for i in tqdm(range(N_TRAIN, N_TRAIN+N_TEST)):
    sil_sample_idx = Y_silenceIdx[i]
    
    y_mulaw = Y_mulaw[i]
    y_mulaw = y_mulaw[:sil_sample_idx.max()+1]
    y_len   = y_mulaw.shape[0]
    
    x_linguistic = X_linguistic[i]
    x_pyworld    = X_pyworld[i]    
    x_melmfcc    = X_melmfcc[i]    
    
    # Feature scaling
#    x_linguistic = minmax_scale(x_linguistic, scale_factors['linguistic_min'], scale_factors['linguistic_max'], feature_range=(0.01, 0.99))
#    x_pyworld    = scale(x_pyworld, 0, scale_factors['pyworld_std'])
#    x_melmfcc    = scale(x_melmfcc, 0, scale_factors['melmfcc_std'])  
    
    # Resampling fs200(5ms-hop) to fs16000
    x_linguistic = librosa.core.resample(x_linguistic.T, 200, args.sr, res_type='kaiser_fast', fix=True, scale=False).T    
    x_pyworld    = librosa.core.resample(x_pyworld.T, 200, args.sr, res_type='kaiser_fast', fix=True, scale=False).T 
    x_melmfcc    = librosa.core.resample(x_melmfcc.T, 200, args.sr, res_type='kaiser_fast', fix=True, scale=False).T 
    
    # Reduce unlabeled index
    x_linguistic = x_linguistic[:sil_sample_idx.max()+1]
    x_pyworld = x_linguistic[:sil_sample_idx.max()+1]
    x_melmfcc = x_linguistic[:sil_sample_idx.max()+1]
    
    # For test, Remove silence samples
    y_mulaw = np.delete(y_mulaw, sil_sample_idx, axis=0)
    x_linguistic = np.delete(x_linguistic, sil_sample_idx, axis=0)
    x_pyworld = np.delete(x_pyworld, sil_sample_idx, axis=0)
    x_melmfcc = np.delete(x_melmfcc, sil_sample_idx, axis=0)
    
    # Save slices (hop=2500, win=5000)
    fname = '{0:012d}.npy'.format(i) # 000000000000.npy, 000000000001.npy, ...
    
    fpath = DST_ROOT + '/TEST/mulaw/' + fname # duplicating '/' is ok.
    np.save(fpath, y_mulaw)
    fpath = DST_ROOT + '/TEST/linguistic/' + fname 
    np.save(fpath, x_linguistic.astype(np.float16))
    fpath = DST_ROOT + '/TEST/pyworld/' + fname 
    np.save(fpath, x_pyworld.astype(np.float16))
    fpath = DST_ROOT + '/TEST/melmfcc/' + fname 
    np.save(fpath, x_melmfcc.astype(np.float16))


#%% Function def.
def displayFeat(x_spec=np.ndarray):
    import matplotlib.pyplot as plt
    import librosa.display    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(x_spec, x_axis='time')
    return 0

##%%
#from nnmnkwii.datasets import FileDataSource, PaddedFileSourceDataset
#from nnmnkwii.datasets import MemoryCacheFramewiseDataset
#from nnmnkwii.preprocessing import trim_zeros_frames, remove_zeros_frames
#from nnmnkwii.preprocessing import minmax, meanvar, minmax_scale, scale
#from nnmnkwii import paramgen
#from nnmnkwii.io import hts
#from nnmnkwii.frontend import merlin as fe
#from nnmnkwii.postfilters import merlin_post_filter
#
#from os.path import join, expanduser, basename, splitext, basename, exists
#import os
#from glob import glob
#import numpy as np
#from scipy.io import wavfile
#from sklearn.model_selection import train_test_split
#import pyworld
#import pysptk
#import librosa
#
#DATA_ROOT = "./data/slt_arctic_full_data"
#test_size = 0.112
#random_state = 1234
#
#
##%%
#mgc_dim = 180
#lf0_dim = 3
#vuv_dim = 1
#bap_dim = 3
#
#duration_linguistic_dim = 416
#acoustic_linguisic_dim = 420 #425
#duration_dim = 5
#acoustic_dim = mgc_dim + lf0_dim + vuv_dim + bap_dim
#
#fs = 16000
#frame_period = 5
#hop_length = 80
#fftlen = 1024
#alpha = 0.41
#
#mgc_start_idx = 0
#lf0_start_idx = 180
#vuv_start_idx = 183
#bap_start_idx = 184
#
#windows = [
#    (0, 0, np.array([1.0])),
#    (1, 1, np.array([-0.5, 0.0, 0.5])),
#    (1, 1, np.array([1.0, -2.0, 1.0])),
#]
##%%
#class BinaryFileSource(FileDataSource):
#    def __init__(self, data_root, dim, train):
#        self.data_root = data_root
#        self.dim = dim
#        self.train = train
#    def collect_files(self):
#        files = sorted(glob(join(self.data_root, "*.bin")))
#        files = files[:len(files)-5] # last 5 is real testset
#        train_files, test_files = train_test_split(files, test_size=test_size,
#                                                   random_state=random_state)
#        if self.train:
#            return train_files
#        else:
#            return test_files
#    def collect_features(self, path):
#        return np.fromfile(path, dtype=np.float32).reshape(-1, self.dim)
#    
##%% This code is only required for calculating max utt lengths. 
#X = {}
#Y = {}
#utt_lengths = {}
#
#for phase in ["train", "test"]:
#    train = phase == "train"
#    x_dim = acoustic_linguisic_dim
#    y_dim = acoustic_dim
#    X[phase] = FileSourceDataset(BinaryFileSource(join(DATA_ROOT, "X_{}".format('acoustic')),
#     dim=x_dim,
#     train=train))
#    Y[phase] = FileSourceDataset(BinaryFileSource(join(DATA_ROOT, "Y_{}".format('acoustic')),
#     dim=y_dim,
#     train=train))
#    # this triggers file loads, but can be neglectable in terms of performance.
#    utt_lengths[phase] = [len(x) for x in X[phase]]
#    
##%% For mini-batching in Pytorch, we use PaddedFileSourceDataset class
#X = {}
#Y = {}
#
#for phase in ["train", "test"]:
#    train = phase == "train"
#    x_dim = acoustic_linguisic_dim
#    y_dim = acoustic_dim
#    X[phase] = PaddedFileSourceDataset(BinaryFileSource(join(DATA_ROOT, "X_{}".format('acoustic')),
#     dim=x_dim, train=train), np.max(utt_lengths[phase]))
#    Y[phase] = PaddedFileSourceDataset(BinaryFileSource(join(DATA_ROOT, "Y_{}".format('acoustic')),
#     dim=y_dim, train=train), np.max(utt_lengths[phase]))
#
#print("Total number of utterances:", len(utt_lengths["train"]))
#print("Total number of frames:", np.sum(utt_lengths["train"]))
#%pylab inline
#rcParams["figure.figsize"] = (16,5)
#hist(utt_lengths["train"], bins=64);