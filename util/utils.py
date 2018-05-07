#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py
Created on Thu May  3 19:32:13 2018

@author: sungkyun
"""
def mu_law_encode(raw_x, num_bins=256):
    import numpy as np
    # -1 < raw_x < 1
    alpha = num_bins - 1 # 0-255
    x_t = np.sign(raw_x) * np.log1p(1 + alpha * np.abs(raw_x)) / np.log1p(1 + num_bins)    
    '''
    Why don't we convert to one-hot vector here?
    => We will use nn.Embedding after, as suggested in:
    https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    '''
    return ((x_t + 1) / 2 * alpha + 0.5).astype(int) 

def mu_law_decode(int_x, num_bins=256):
    import numpy as np
    alpha = num_bins - 1
    x_rescale = 2 * (int_x.astype(float) / alpha) - 1 # [0,255] --> [-1,1]  
    magnitude = (1 / alpha) * ((1 + alpha)**abs(x_rescale) - 1)    
    return np.sign(x_rescale) * magnitude


