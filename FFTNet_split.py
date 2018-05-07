#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 21:47:37 2018

@author: sungkyun
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Models with Preset (for convenience)
'''
dim_input: dimension of input (256 for 8-bit mu-law input)
num_layer: number of layers (11 in paper). receptive field = 2^11 (2,048)
io_ch: number of input(=output) channels in each fft layers 
skip_ch: number of skip-channels, only required for fft-residual net.

Annotations:
    B: batch dimension
    C: channel dimension
    L: length dimension
'''
def fftnet_base(input_dim=256, num_layer=11, io_ch=256): 
    return FFTNet(input_dim=input_dim, num_layer=num_layer, io_ch=io_ch, skip_ch=0, bias=True)

def fftnet_residual(input_dim=256, num_layer=11, io_ch=256, skip_ch=256):   
    return FFTNet(input_dim=input_dim, num_layer=num_layer, io_ch=io_ch, skip_ch=skip_ch, bais=True)


# FFT_Block: define a basic FFT Block
'''
FFT_Block: 
    - described in the paper, section 2.2.
    - in case of the first layer used in the first FFT_Block, 
      we use nn.embedding layer for one-hot index(0-255) entries.  
'''
class FFT_Block(nn.Module):
       
    def __init__(self, is_first_block=True, initial_input_dim=256, cond_dim=25, io_ch=int, recep_sz=int, bias=True):
        
        super(FFT_Block, self).__init__()
        self.is_first_block = is_first_block     # If True, an input_embedding_layer will be created (this is not clear in the paper).
        self.initial_input_dim=initial_input_dim # This argument is only required for constructing the first block with one-hot input.
        self.cond_dim=cond_dim                   # Number of dimensions of condition input
        self.io_ch = io_ch
        self.recep_sz = recep_sz                 # Size of receptive field: i.e., the 1st layer has receptive field of 2^11(=2,048). 2nd has 2^10. 
        self.bias=bias                           # If True, use bias in 1x1 conv.

        # NN Modules:
        if self.is_first_block is True:
            self.input_embedding_layer = nn.Embedding(num_embeddings=initial_input_dim, embedding_dim=io_ch) # one-hot_index -> embedding -> 256ch output 
        self.conv_1x1_L = nn.Conv1d(in_channels=self.io_ch, out_channels=self.io_ch, kernel_size=1, stride=1, bias=self.bias)
        self.conv_1x1_R = nn.Conv1d(in_channels=self.io_ch, out_channels=self.io_ch, kernel_size=1, stride=1, bias=self.bias)
        self.conv_1x1_VL = nn.Conv1d(in_channels=self.cond_dim, out_channels=self.io_ch, kernel_size=1, stride=1, bias=self.bias)
        self.conv_1x1_VR = nn.Conv1d(in_channels=self.cond_dim, out_channels=self.io_ch, kernel_size=1, stride=1, bias=self.bias)
        self.conv_1x1_last = nn.Conv1d(in_channels=self.io_ch, out_channels=self.io_ch, kernel_size=1, stride=1, bias=self.bias)
        
        return None
    
    
    def split_tensor(self, x):      # a tensor x with size(BxCxL)
        '''
        See more details in 'Explanation_of_sum(split_1x1_conv).md' file. 
        In summary, the required paddings and omissions for preparing split_1x1_conv are: 
            
                <First Block>
                L: left_zpad(input, recep_sz), right_omit(recep_sz/2)
                R: left-zpad(recep_sz/2)
                
                <Normal Blocks>
                recep_sz = recep_sz/2  <-- size of halves decreases to half.
                L: right_omit(recep_sz/2)
                R: left_omit(recep_sz/2)
                
                <Final stage before FC layer>
                right_omit(1)
        ''' 
        if self.is_first_block is True:
            x_L = F.pad(x, (self.recep_sz, 0), 'constant', 0)        # left-padding with zeros
            x_L = x_L[:, :, 0:x_L.shape[2] - int(self.recep_sz/2)]   # right-omit
            x_R = F.pad(x, (int(self.recep_sz/2), 0), 'constant', 0) # left-padding with zeros
        else: # Normal blocks...
            x_L = x[:, :, 0:x.shape[2] - int(self.recep_sz/2)]     # right-omit
            x_R = x[:, :, int(self.recep_sz/2):]                   # left-omit
            
        return x_L, x_R
        
        
    def forward(self, x, cond):
        
        if self.is_first_block is True:
            x = self.input_embedding_layer(x)                   # In : BxL, Out: BxLxC
            x = x.permute(0,2,1)                                # Out: BxCxL
        
        # Split input x into 2 halves, then 1x1 Conv
        x_L, x_R = self.split_tensor(x)
        z = self.conv_1x1_L(x_L) + self.conv_1x1_R(x_R)         # Eq(1), z = w_L*x_L + w_R*x_R

        # Adding auxiliary condition as Eq(2) in paper.
        h_L, h_R = self.split_tensor(cond)                      # Split condition input into left and right
        z = z + self.conv_1x1_VL(h_L) + self.conv_1x1_VR(h_R)   # Eq(2), z = (WL ∗ xL + WR ∗ xR) + (VL ∗ hL + VR ∗ hR)
        x = F.relu(self.conv_1x1_last(F.relu(z)))               # x = ReLU(conv1x1(ReLU(z)))
        
        # Zero-padding for cond is required for next layer. 
        return x
    
'''
FFTNet: 
    - [11 FFT_blocks] --> [FC_layer] --> [softmax] 
'''
class FFTNet(nn.Module):
    def __init__(self, input_dim=256, cond_dim=25, num_layer=11, io_ch=256, skip_ch=0, bias=True ):
        
        super(FFTNet, self).__init__()
        self.input_dim = input_dim                       # 256 (=num_classes)
        self.cond_dim = cond_dim                         # 25
        self.num_layer = num_layer                       # 11
        self.io_ch = io_ch
        self.skip_ch = skip_ch
        self.bias = bias
        self.max_recep_sz = int(pow(2, self.num_layer))  # 2^11
        
        # Constructing FFT Blocks:
        blocks = nn.ModuleList()
        for l in range(self.num_layer):
            if l is 0: # First block 
                recep_sz = self.max_recep_sz       # 2^11 = 2048
                blocks.append( FFT_Block(is_first_block=True, 
                                         initial_input_dim=self.input_dim,
                                         cond_dim=self.cond_dim,
                                         io_ch=self.io_ch,
                                         recep_sz=recep_sz,
                                         bias=self.bias) )
            else:
                recep_sz = int(pow(2, self.num_layer-l))     # 1024, 512, ... 2
                blocks.append( FFT_Block(is_first_block=False,
                                         cond_dim=self.cond_dim,
                                         io_ch=self.io_ch,
                                         recep_sz=recep_sz,
                                         bias=self.bias) )
        self.fft_blocks=blocks 
        
        # Final FC layer: 
        self.fc = nn.Linear(in_features=self.io_ch, out_features=self.io_ch)
        
        return None
    
    
    def forward(self, x, cond):
        
        for l in range(self.num_layer):
            if l is 0:
                x = self.fft_blocks[l](x, cond)
            else:
                zpad_sz = int(self.max_recep_sz/pow(2, l))
                padded_cond = F.pad(cond, (zpad_sz, 0), 'constant', 0)
                x = self.fft_blocks[l](x, padded_cond)
        
        x = x[:,:,:-1]        # right-omit 1 is required.
        x = x.permute(0,2,1)  # (BxCxL) --> (BxLxC)
        x = self.fc(x)        # (BxLxC)
        # NOTE: in torch, softmax() is included in CE loss.
        
        return x