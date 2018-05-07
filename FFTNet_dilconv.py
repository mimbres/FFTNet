#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:46:10 2018

@author: sungkyun

FFTNet model using 2x1 dil-conv
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
    - using 2x1 dilated-conv, instead of LR split 1x1 conv.
    - described in the paper, section 2.2.
    - in case of the first layer used in the first FFT_Block, 
      we use nn.embedding layer for one-hot index(0-255) entries.  
'''
class FFT_Block(nn.Module):
       
    def __init__(self, cond_dim=25, io_ch=int, recep_sz=int, bias=True):
        
        super(FFT_Block, self).__init__()
        self.cond_dim=cond_dim                   # Number of dimensions of condition input
        self.io_ch = io_ch
        self.recep_sz = recep_sz                 # Size of receptive field: i.e., the 1st layer has receptive field of 2^11(=2,048). 2nd has 2^10. 
        self.bias = bias                           # If True, use bias in 1x1 conv.
        self.dilation = int(recep_sz / 2)

        self.conv_2x1_LR = nn.Conv1d(in_channels=self.io_ch, out_channels=self.io_ch, 
                                     kernel_size=2, stride=1, dilation=self.dilation, bias=self.bias)
        self.conv_2x1_VLR = nn.Conv1d(in_channels=self.cond_dim, out_channels=self.io_ch, 
                                     kernel_size=2, stride=1, dilation=self.dilation, bias=self.bias)
        self.conv_1x1_last = nn.Conv1d(in_channels=self.io_ch, out_channels=self.io_ch, 
                                       kernel_size=1, stride=1, bias=self.bias)
        return None
 
    
    def forward(self, x, cond):

        z = self.conv_2x1_LR(x)                     # Eq(1), z = w_L*x_L + w_R*x_R
        z = z + self.conv_2x1_VLR(cond)             # Eq(2), z = (WL ∗ xL + WR ∗ xR) + (VL ∗ hL + VR ∗ hR)
        x = F.relu(self.conv_1x1_last(F.relu(z)))   # x = ReLU(conv1x1(ReLU(z)))

        return x


'''
FFTNet: 
    - [11 FFT_blocks] --> [FC_layer] --> [softmax] 
'''
class FFTNet(nn.Module):
    def __init__(self, input_dim=256, cond_dim=25, num_layer=11, io_ch=256, skip_ch=0, bias=True):
        
        super(FFTNet, self).__init__()
        self.input_dim = input_dim                       # 256 (=num_classes)
        self.cond_dim = cond_dim                         # 25
        self.num_layer = num_layer                       # 11
        self.io_ch = io_ch                               # 256 ch. in the paper
        self.skip_ch = skip_ch                           # Not implemented yet (no skip channel in the paper)
        self.bias = bias                                 # If True, use bias in 2x1 conv.
        self.max_recep_sz = int(pow(2, self.num_layer))  # 2^11, max receptive field size
        
        # Embedding layer: one-hot_index -> embedding -> 256ch output 
        self.input_embedding_layer = nn.Embedding(num_embeddings=self.input_dim, 
                                                  embedding_dim=self.io_ch) 
                
        # Constructing FFT Blocks:
        blocks = nn.ModuleList()
        for l in range(self.num_layer):
            recep_sz = int(pow(2, self.num_layer-l))     # 1024, 512, ... 2
            blocks.append( FFT_Block(cond_dim=self.cond_dim,
                                     io_ch=self.io_ch,
                                     recep_sz=recep_sz,
                                     bias=self.bias) )
        self.fft_blocks=blocks 
        
        # Final FC layer: 
        self.fc = nn.Linear(in_features=self.io_ch, out_features=self.io_ch)
        
        return None
    
    
    def forward(self, x, cond, gen_mod=False):
        
        # Padding x:
        zpad_sz = int(self.max_recep_sz)
        x = F.pad(x, (zpad_sz, 0), 'constant', 128)         # 128? or 0?
        
        # Embedding(x):
        x = self.input_embedding_layer(x)                   # In : BxL, Out: BxLxC
        x = x.permute(0,2,1)                                # Out: BxCxL
        
        # FFT_Blocks:
        for l in range(self.num_layer):
            # Padding cond:
            zpad_sz = int(self.max_recep_sz/pow(2, l))
            padded_cond = F.pad(cond, (zpad_sz, 0), 'constant', 0)
         
            x = self.fft_blocks[l](x, padded_cond)
   
        if gen_mod is True:
            x = x[:,:,-1]         # In generator mode, take the last one sample only.
            x = x.reshape(-1, 1, self.io_ch) # (BxC) --> (Bx1xC)
        else:
            x = x[:,:,:-1]        # In training mode, right-omit 1 is required.
            x = x.permute(0,2,1)  # (BxCxL) --> (BxLxC)
            
        x = self.fc(x)        # (BxLxC)
        # NOTE: in PyTorch, softmax() is included in CE loss.
        
        return x
    