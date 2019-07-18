#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:13:46 2018

@author: avelinojaver

"""

from .unet import conv_block, init_weights, calculate_padding, crop_to_match

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

#base on https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py ...
#... and https://github.com/ozan-oktay/Attention-Gated-Networks/blob/eed71108da598cba69ab4c14dac2fdc0688516c0/models/layers/grid_attention_layer.py

class AttentionBlock(nn.Module):
    def __init__(self, 
                 gating_channels, 
                 in_channels,
                 subsample_factor = 2,
                 batchnorm = True
                 ):
        super().__init__()
        
        inter_channels = in_channels // 2
        
        _layers = [nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)]
        if batchnorm:
            _layers.append(nn.BatchNorm2d(inter_channels))
        self.W_gating = nn.Sequential(*_layers)
        
        
        _layers = [nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=subsample_factor, padding=0, bias=True)]
        if batchnorm:
            _layers.append(nn.BatchNorm2d(inter_channels))
        self.W_input = nn.Sequential(*_layers)
        
        
        _layers = [nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)]
        if batchnorm:
            _layers.append(nn.BatchNorm2d(in_channels))
        self.W_output = nn.Sequential(*_layers)
        
        _layers = [
                    nn.ReLU(),
                    nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
                    ]
        if batchnorm:
            _layers.append(nn.BatchNorm2d(1))
        _layers.append(nn.Sigmoid())
        
        self.psi = nn.Sequential(*_layers)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x_in):
        x1 = self.W_input(x_in)
        
        g1 = self.W_gating(g)
        g1 = F.interpolate(g1, 
                          size = x1.shape[2:],
                          mode = 'bilinear'
                          )
        
        psi = self.psi(g1 + x1)
        
        psi = F.upsample(psi, 
                         size = x_in.shape[2:], 
                         mode='bilinear')
        
        
        y = self.W_output(psi * x_in)
        return y

class Down(nn.Module):
    def __init__(self, 
                 n_filters,
                 downsampling_mode = 'maxpool', 
                 n_conv3x3 = 1, 
                 **argkws):
        super().__init__()
        
        if downsampling_mode == 'maxpool':
            downsampling_l = nn.MaxPool2d(2)
        else:
            ValueError('Downsampling mode `{}` not implemented.'.format(downsampling_mode))
        
        _layers = []
        for ii in range(len(n_filters) - 1):
            n_in, n_out = n_filters[ii], n_filters[ii+1]
            _layers += conv_block(n_in, n_out, **argkws)
        
            
        _layers.append(downsampling_l)
        
        self.conv_pooled = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv_pooled(x)
        return x


class UpAttention(nn.Module):
    def __init__(self, 
                 n_filters, 
                 interp_mode = 'bilinear', 
                 batchnorm = False,
                 **argkws):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        
        self.interp_mode = interp_mode
        
        attn_filter = n_filters[0] // 2
        self.attn = AttentionBlock(attn_filter, attn_filter, batchnorm = batchnorm)
        
        _layers = []
        for ii in range(len(n_filters) - 1):
            n_in, n_out = n_filters[ii], n_filters[ii+1]
            _layers += conv_block(n_in, n_out, batchnorm = batchnorm, **argkws)
        
        
        self.conv = nn.Sequential(*_layers)

    def forward(self, x2, x1):
        
        x1_gated = self.attn(x2, x1)
        
        x2_interp = F.interpolate(x2, 
                           scale_factor = 2,
                           mode = self.interp_mode
                           )
        
        x1_crop = crop_to_match(x2_interp, x1_gated)
        
        x = torch.cat([x1_crop, x2_interp], dim=1)
        x = self.conv(x)
        return x

class UNetBase(nn.Module):
    def __init__(self, 
                 DownBlock,
                 UpBlock,
                 n_channels = 1, 
                 n_classes = 1, 
                 batchnorm = False,
                 init_type = 'xavier',
                 pad_mode = 'constant'
                 ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.pad_mode = pad_mode
        
        
        _layers = conv_block(n_channels,
                               64, 
                               kernel_size = 7, 
                               batchnorm = batchnorm, 
                               padding = 3)
        self.conv0 = nn.Sequential(*_layers)
        
        self.down1 = DownBlock([64, 128, 128])
        self.down2 = DownBlock([128, 256, 256])
        self.down3 = DownBlock([256, 512, 512])
        self.down4 = DownBlock([512, 512, 512])
        
        
        
        
        self.up4 = UpBlock([1024, 256, 256])
        self.up3 = UpBlock([512, 128, 128])
        self.up2 = UpBlock([256, 128, 64])
        self.up1 = UpBlock([128, 32, 32])
        
        
        dd = nn.Conv2d(32, n_classes, 3, padding = 1)
        self.conv_out = nn.Sequential(dd)
        
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
    
    def _unet(self, x_input):    
        x0 = self.conv0(x_input)
        
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        
        
        x = self.up4(x4, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)
        
        x = self.conv_out(x)
        
        
        return x
    
    
    
    def forward(self, x_input):
        pad_, pad_inv_ = calculate_padding(x_input.shape[2:])
        
        x_input = F.pad(x_input, pad_, mode = self.pad_mode)
        x = self._unet(x_input)
        x = F.pad(x, pad_inv_)
        
        return x
    

class UNetAttention(UNetBase):
    def __init__(self,
                 *args,
                 batchnorm = False,
                 interp_mode = 'nearest',
                 **argkws
                 ):
        
        _down_func = partial(Down, batchnorm = batchnorm)
        
        _up_func = partial(UpAttention, 
                           interp_mode = interp_mode,
                           batchnorm = batchnorm
                           )
        
        
        super().__init__(_down_func, _up_func, *args, batchnorm = batchnorm, **argkws)
        
#%%
if __name__ == '__main__':
    n_in = 3
    n_out = 2
    batchnorm = True
    interp_mode = 'nearest'
    
    mod = UNetAttention(n_in, 
               n_out, 
               batchnorm = batchnorm,
               interp_mode = interp_mode,
               )
    
    im_size  = 196, 196
    X = torch.rand((1, n_in, *im_size))
    target = torch.rand((1, n_out, *im_size))
    
    
    out = mod(X)
    
    #if mod.valid_padding:
        #pad_, pad_inv_ = mod._calculate_padding(X)
        #target = 
    
    loss = (out-target).abs().sum()
    loss.backward()
    
    print(out.size())