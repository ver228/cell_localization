#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:13:46 2018

@author: avelinojaver

"""

from .unet_base import UpSimple, DownSimple, unet_constructor

import torch
from torch import nn
import torch.nn.functional as F
import math

__all__ = ['unet_attention']

class AttentionBlock(nn.Module):
    #base on https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py ...
    #... and https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eed71108da598cba69ab4c14dac2fdc0688516c0/models/layers/grid_attention_layer.py

    def __init__(self, 
                 gating_channels, 
                 in_channels,
                 subsample_factor = 2,
                 batchnorm = True
                 ):
        super().__init__()
        
        inter_channels = in_channels // 2 # the number of inter_channels is not that important since at the end psi output will be [n_batchx1xHxW]
        Wg_layers = [nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)]
        if batchnorm:
            Wg_layers.append(nn.BatchNorm2d(inter_channels))
        self.W_gating = nn.Sequential(*Wg_layers)
        
        
        Wi_layers = [nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=subsample_factor, padding=0, bias=True)]
        if batchnorm:
            Wi_layers.append(nn.BatchNorm2d(inter_channels))
        self.W_input = nn.Sequential(*Wi_layers)
        
        
        psi_layers = [nn.ReLU(),
                    nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
                    ]
        if batchnorm:
            psi_layers.append(nn.BatchNorm2d(1))
        psi_layers.append(nn.Sigmoid())
        self.psi = nn.Sequential(*psi_layers)
        
        
        Wout_layers = [nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)]
        if batchnorm:
            Wout_layers.append(nn.BatchNorm2d(in_channels))
        self.W_output = nn.Sequential(*Wout_layers)
        
        
    def forward(self, g, x_in):
        x1 = self.W_input(x_in)
        
        g1 = self.W_gating(g)
        g1 = F.interpolate(g1, 
                          size = x1.shape[2:],
                          mode = 'bilinear',
                          align_corners=False
                          )
        
        psi = self.psi(g1 + x1)
        
        psi = F.interpolate(psi, 
                         size = x_in.shape[2:], 
                         mode='bilinear',
                         align_corners=False
                         )
        
        
        y = self.W_output(psi * x_in)
        
        return y

class UpAttention(UpSimple):
    def __init__(self, *args, attn_gating_channels = None, attn_in_channels = None, batchnorm = False, **argkws):
        if attn_gating_channels is None:
            raise ValueError(f'Please give a valid value for attention filters.')
        
        super().__init__(*args, **argkws)
        
        self.attn = AttentionBlock(attn_gating_channels, attn_in_channels, batchnorm = batchnorm)
        

    def forward(self, x1, x2):
        x2_gated = self.attn(g = x1, x_in = x2)
        x = super().forward(x1, x2_gated)
        return x

def unet_attention(n_in, n_out, increase_factor = 2, **argkws):
    
    
    def UpAttentionS(n_filters, **argkws):
        #I am adding this function to automatically get the number of expected filters between layers.
        #Since I am calculating this automatically in `unet_constructor` I just need `factor` and the current number of filters
        
        attn_in_channels = int(math.ceil((n_filters[0])/(increase_factor + 1)))
        attn_gating_channels = n_filters[0] - attn_in_channels
        return UpAttention(n_filters, 
                           attn_in_channels = attn_in_channels, 
                           attn_gating_channels = attn_gating_channels,
                           **argkws)
    
    model = unet_constructor(n_in, n_out, DownSimple, UpAttentionS, increase_factor = increase_factor, **argkws)
    return model

 #%%      
if __name__ == '__main__':
    n_in = 3
    n_out = 2
    batchnorm = True
    im_size  = 196, 196
    X = torch.rand((1, n_in, *im_size))
    target = torch.rand((1, n_out, *im_size))
    
    model = unet_attention(n_in, n_out, batchnorm = batchnorm, factor = 2, initial_filter_size = 32)
    out = model(X)
    
    loss = (out-target).abs().sum()
    loss.backward()
    
    print(out.size())
