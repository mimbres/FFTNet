#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 22:37:27 2018

@author: sungkyun
"""

X_mulaw = Variable(torch.LongTensor(4,3000)*0, volatile=True)              # input: 4 x 3000 (batch x T) 
input_layer = nn.Embedding(num_embeddings=256, embedding_dim=256)  # target dim=10
out = input_layer(X_mulaw) 
input=out.permute(0,2,1)
input=input[:,:,:8]
x = input

input = F.pad(x, (8, 0), 'constant', 0)   

#<block 1>
recep_sz=8
dilation=int(recep_sz/2)
conv_1x1_LR = nn.Conv1d(in_channels=256, out_channels=256,
                        kernel_size=2, stride=1, dilation=4, bias=False)

q=conv_1x1_LR(input)

#<block 2>
recep_sz=4
dilation = 2
conv_1x1_LR = nn.Conv1d(in_channels=256, out_channels=256,
                        kernel_size=2, stride=1, dilation=2, bias=False)
qq = conv_1x1_LR(q)



#%% Unit-test for FFT_Blocks

#nput_onehot_idx = Variable(torch.LongTensor([0,1,2,1,2,3,2,0]))  # assuming 2-bit encoded mu_law_x 
X_mulaw = Variable(torch.LongTensor(4,3000)*0, volatile=True)              # input: 4 x 3000 (batch x T) 
X_mfcc = Variable(torch.FloatTensor(4, 25, 3000), volatile=True)
input_layer = nn.Embedding(num_embeddings=256, embedding_dim=256)  # target dim=10
out = input_layer(X_mulaw)                                        # output:4 x 3000 x 10 (BxLxC)
        



#%% 1x1 conv test
out_channels = 3
in_channels = 5
iW = 8
input = Variable(torch.FloatTensor(1,in_channels, iW)) # 1x5x8

kW = 1   # kernel size
conv_0 = torch.nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kW,
                         stride=1, padding=0, dilation=1, groups=1, bias=True)
conv_0(input) # 1x3x8



#%% 
for batch_idx, (_, X_mulaw, X_mfcc ) in enumerate(train_loader):
    X_mulaw = Variable(X_mulaw)
    X_mfcc = Variable(X_mfcc.float())
    
    
#%%
%pylab inline
import matplotlib
import seaborn
seaborn.set_style("dark")
rcParams['figure.figsize'] = (16, 6)

    