#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 21:11:57 2018
@author: sungkyun

FFTNet: A REAL-TIME SPEAKER-DEPENDENT NEURAL VOCODER (Zeyu Jin et al., 2018, ICASSP)

*CMU Arctic dataset:
    - 1032 utterances for train
    - 100 utterances for test
    - cond. input = F0 and 25-dim MCC
    
*Preprocessing:
    - 16Khz sampling mono audio
    - 8-bit mu-law encoded wav

*Architecture:
    - receptive field = 2048 (2^11)
    - 11 FFT-layers (256 ch.)
    - Total 1M parameters 
    - final 2 layers =  FC() -> softmax(256) : see details in paper 2.3.2

*Training:
    - minibatch of 5 x 5000 samples
    - 100,000 steps = 500 iters

*Requirements:
    pip install soundfile
    pip install librosa

"""
import os
import argparse
import numpy as np
import torch
from torch.nn import DataParallel 
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
#from util.loading_dataset import YesNoDataset # required for loading YesNo dataset
from util.live_loading_dataset import CmuArcticLiveDataset
from util.utils import mu_law_decode # required for auditioning generated samples


# Parsing arguments
parser = argparse.ArgumentParser(description='FFTNet implementation')
parser.add_argument('-exp', '--exp_name', type=str, default='00', metavar='STR',
                    help='Generated samples will be located in the checkpoints/exp<exp_name> directory. Default="00"') # 
parser.add_argument('-e', '--max_epoch', type=int, default=10000, metavar='N',
                    help='Max epoch, Default=10000') 
parser.add_argument('-btr', '--batch_train', type=int, default=5, metavar='N',
                    help='Batch size for training. e.g. -btr 5')
parser.add_argument('-bts', '--batch_test', type=int, default=1, metavar='N',
                    help='Batch size for test. e.g. -bts 5')
parser.add_argument('-load', '--load', type=str, default=None, metavar='STR',
                    help='e.g. --load checkpoints/exp00/checkpoint_00')
parser.add_argument('-sint', '--save_interval', type=int, default=10, metavar='N',
                    help='Save interval., default=100')
parser.add_argument('-g', '--gpu_id', type=str, default=None, metavar='STR',
                    help='Multi GPU ids to use')
args = parser.parse_args()

USE_GPU = torch.cuda.is_available()
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
    
RAND_SEED  = 0

#%% Loading data
dset_train = CmuArcticLiveDataset(train_flag=True, cond_sel='mfcc', cache_size=1000) # wav is cahced in memory. 
dset_test = CmuArcticLiveDataset(train_flag=False, cond_sel='mfcc', cache_size=1)


train_loader = DataLoader(dset_train,
                          batch_size=args.batch_train,
                          shuffle=True,
                          num_workers=6,
                          pin_memory=True
                         ) # number of CPU threads, practically, num_worker = 4 * num_GPU

test_loader = DataLoader(dset_test,
                          batch_size=args.batch_test,
                          shuffle=False,
                          num_workers=8,
                          pin_memory=True,
                         )


    
#%% Train & Test Functions
def train(epoch):
    model.train()
    train_loss = 0.
    train_acc = []
    total_data_sz = train_loader.dataset.__len__()
    
    for batch_idx, (_, X_mulaw, X_mfcc ) in enumerate(train_loader):
        if USE_GPU:
            #X_mulaw, X_mfcc = X_mulaw.cuda(), X_mfcc.cuda()
            X_mulaw, X_mfcc = Variable(X_mulaw.cuda()), Variable(X_mfcc.cuda().float())
            #X_mulaw, X_mfcc = Variable(X_mulaw.long()), Variable(X_mfcc.half()) # only for fp16 supported GPU!
        else:
            X_mulaw, X_mfcc = Variable(X_mulaw), Variable(X_mfcc.float())
        
        optimizer.zero_grad()
        y = model(X_mulaw, X_mfcc)
        loss = F.cross_entropy(input=y.view(-1, 256), target=X_mulaw.view(-1), size_average=True) 
        # input=y.view(-1, num_classes), where num_classes=256 for 8bit mu-law
        # target=X_mulaw.view(-1)
        loss.backward()
        optimizer.step()
        
        # Accuracy
        pred = y.view(-1,256).data.max(1, keepdim=True)[1] # Get the index of the max log-probability from y
        acc = pred.eq(X_mulaw.view(-1).data.view_as(pred)).cpu().sum().numpy() / len(pred) # Compute accuracy by comparing pred with X_mulaw 
        print('Train Epoch: {} [{}/{}],   Loss = {:.6f},   Acc = {:.6f}'.format(
                epoch, batch_idx * train_loader.batch_size, total_data_sz, loss.item(), acc))
        
        train_loss += loss.item()
        train_acc.append(acc)
        
    return train_loss, np.mean(train_acc)




def generator(test_file_id, out_filename, recep_sz=2048, verbose=1000):
    '''
    Annotation
        - X_mfcc        : Original condition input
        - total samples : The number of total samples to generate (=size(X_mfcc))
        - x_mulaw_slice : A temporary input replacing X_mulaw. At the start, all zeros(encoded as 128s). 
        - x_mfcc_slice  : A temporary condition input.
        - y             : One sample output of the model. This will be fed into x_mulaw_slice at the right-most column.
        - pred          : Prediction of one new generated sample
        - out           : Collection of mu_law_decode(pred)s.
    '''
    _, _, X_mfcc  = test_loader.dataset.__getitem__(test_file_id) # X_mfcc: (CxL) np.ndarray
    feat_dim      = X_mfcc.shape[0]
    total_samples = X_mfcc.shape[1]
    X_mfcc = X_mfcc.reshape(1, feat_dim, total_samples)  # BxCxL
    
    # Initial input slices, filled with zeros.
    x_mulaw_slice = Variable(torch.LongTensor(1, recep_sz) * 0 + 128, requires_grad=False).cuda() # all zeros(128s).
    x_mfcc_slice  = Variable(torch.FloatTensor(1, feat_dim, recep_sz) * 0., requires_grad=False).cuda()
    out           = []
    
    model.eval()  # .eval(): Not requires gradients.
    
    for i in range(total_samples):
        # New x_mfcc_slice: shift-left, then fill one 'cond' sample into the right-most column
        x_mfcc_slice[:, :, 0:-1] = x_mfcc_slice[:, :, 1:]
        x_mfcc_slice[:, :, -1]   = torch.FloatTensor(X_mfcc[:, :, i])
        
        y = model(x_mulaw_slice, x_mfcc_slice, gen_mod=True)  # 1x1x256 (BxLxC)
        pred = y.view(-1,256).data.max(1, keepdim=True)[1]    # Predict: Get the index of the max log-probability from y
        
        # New x_mulaw_slice: shift-left, then fill 'pred' into the right-most column 
        x_mulaw_slice[0, 0:-1] = x_mulaw_slice[0, 1:]         # Shift-left
        x_mulaw_slice[0, -1]   = pred                         # Push 'pred'
        
        # Collect generated sample
        out.append(float(mu_law_decode(pred.cpu().numpy())))
        
        # Print progress
        if i % verbose == 0:
            print('Generator: {}/{} samples ({:.2f}%)'.format(i, total_samples,
                  100 * i / total_samples ) )
    
    # Save audio
    import librosa
    librosa.output.write_wav(out_filename, np.asarray(out), sr=16000)
    
    

def validate(epoch):
    model.eval()
    val_loss = 0.
    # Not implemented yet
    return val_loss


def load_checkpoint(filepath):
    '''
    Load pre-trained model.
    '''
    dt = torch.load(filepath)
    print('Loading from epoch{}...'.format(dt['epoch']) )
    model.load_state_dict(dt['state_dict'])
    optimizer.load_state_dict(dt['optimizer'])
    return dt['epoch']

def save_checkpoint(state, accuracy, exp_name):
    checkpoint_dir = 'checkpoints/' + exp_name  
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = checkpoint_dir + '/checkpoint{}.pth.tar'.format(state['epoch'])
    torch.save(state, filepath)

def history_recorder():
    '''
    history_recorder():
        - save training history as .csv files.
        - save learning-curve as .png files
    '''
    return 0

def print_model_sz(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('Number of trainable parameters = {}'.format(sum([np.prod(p.size()) for p in model_parameters])) )

#%% Experiment: train
from FFTNet_dilconv import FFTNet    # <-- implemented using 2x1 dilated conv. 
#from FFTNet_split   import FFTNet    # <-- same with paper

model = FFTNet(cond_dim=26).cuda() if USE_GPU else FFTNet(cond_dim=26).cpu()  # or .cuda()

# Multi-gpu support
if (args.gpu_id is not None) & (torch.cuda.device_count() > 1):
    model = DataParallel(model).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
print_model_sz(model)

torch.backends.cudnn.benchmark = True

last_epoch = 0
if args.load is not None:
    #load_checkpoint('checkpoints/00/checkpoint.pth.tar')
    last_epoch = load_checkpoint(args.load)

for epoch in range(last_epoch, args.max_epoch):
    torch.manual_seed(RAND_SEED + epoch)
    train_loader.dataset.rand_flush()
    tr_loss, tr_acc = train(epoch)
    
    if (epoch % args.save_interval) is 0:
        save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),},
                tr_acc, args.exp_name)
    

#%% Experiment: generation
test_file_id = 0 # Select 0~5 for different condition input
generator(test_file_id=test_file_id, out_filename='aaa.wav')




    
    
