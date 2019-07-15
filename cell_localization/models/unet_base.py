#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:04:43 2019

@author: avelinojaver
"""
from torch.nn import init
import math

import torch
from torch import nn
import torch.nn.functional as F

def init_weights(net, init_type='xavier', init_gain=0.02):
    """Initialize network weights.
    Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)  # apply the initialization function <init_func>
    




class ConvBlock(nn.Sequential):
    def __init__(self, n_in, n_out, kernel_size = 3, batchnorm = False, padding = None):
        
        if padding is None:
            padding = kernel_size//2
        conv = nn.Conv2d(n_in, n_out, kernel_size = kernel_size, padding = padding)
        act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
        if batchnorm:
            _layers = [conv, nn.BatchNorm2d(n_out),  act]
        else:
            _layers = [conv, act]
            
            
        super().__init__(*_layers)
    
class DownSimple(nn.Module):
    def __init__(self, 
                 n_filters,
                 n_conv3x3 = 1, 
                 **argkws):
        super().__init__()
        
        _layers = []
        for ii in range(len(n_filters) - 1):
            n_in, n_out = n_filters[ii], n_filters[ii+1]
            _layers += ConvBlock(n_in, n_out, **argkws)
        self.conv = nn.Sequential(*_layers)
        
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x
    
def crop_to_match(x, x_to_crop):
    c = (x_to_crop.size()[2] - x.size()[2])/2
    c1, c2 =  math.ceil(c), math.floor(c)
    
    c = (x_to_crop.size()[3] - x.size()[3])/2
    c3, c4 =  math.ceil(c), math.floor(c)
    
    cropped = F.pad(x_to_crop, (-c3, -c4, -c1, -c2)) #negative padding is the same as cropping
    
    return cropped

class UpSimple(nn.Module):
    def __init__(self, 
                 n_filters, 
                 interp_mode = 'nearest', 
                 **argkws):
        super().__init__()
        
        self.interp_mode = interp_mode
        _layers = []
        for ii in range(len(n_filters) - 1):
            n_in, n_out = n_filters[ii], n_filters[ii+1]
            _layers += ConvBlock(n_in, n_out, **argkws)
        self.conv = nn.Sequential(*_layers)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, 
                           scale_factor = 2,
                           mode = self.interp_mode
                           )
        x2 = crop_to_match(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        return x
    
class UNetBase(nn.Module):
    def __init__(self, 
                 InitialBlock,
                DownBlock,
                UpBlock,
                OutputBlock,
                n_inputs,
                down_filters,
                up_filters,
                n_outputs,
                 init_type = 'xavier',
                 **argkws
                 ):
        super().__init__()
        
        first_down_filter = down_filters[0][0] 
        last_up_filter = down_filters[-1][-1]
        
        self.initial_block = InitialBlock(n_inputs, first_down_filter, **argkws)
        self.down_blocks = [DownBlock(*x, **argkws) for x in down_filters]
        self.up_blocks = [UpBlock(*x, **argkws) for x in down_filters]
        self.output_block = OutputBlock(n_inputs, last_up_filter, **argkws)
        
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
    
    def _unet(self, x_input):    
        x = self.initial_block(x_input)
        
        x_downs = []
        for down in self.down_blocks:
            x_downs.append(x)
            x = down(x)
        
        for x_down, up in zip(x_downs[::-1], self.up_blocks):
            x = up(x, x_down)
        
        x = self.output_block(x)
        
        return x
    
    @staticmethod
    def calculate_padding(x_shape):
        # the input shape must be divisible by 32 otherwise it will be cropped due 
        #to the way the upsampling in the network is done. Therefore it is better to path 
        #the image and recrop it to the original size
        nn = 2**5
        ss = [math.ceil(x/nn)*nn - x for x in x_shape]
        pad_ = [(int(math.floor(x/2)),int(math.ceil(x/2))) for x in ss]
        
        #use pytorch for the padding
        pad_ = [x for d in pad_[::-1] for x in d] 
        pad_inv_ = [-x for x in pad_] 
    
        return pad_, pad_inv_
    
    def forward(self, x_input):
        pad_, pad_inv_ = self.calculate_padding(x_input.shape[2:])
        
        x_input = F.pad(x_input, pad_, mode = self.pad_mode)
        x = self._unet(x_input)
        x = F.pad(x, pad_inv_)
        
        return x
    
#%%
def unet_progressive(n_inputs, 
                     n_ouputs, 
                     initial_filter_size = 48, 
                     n_levels = 4, 
                     n_conv_per_level = 2,
                     factor = 2
                     ):
    
    down_filters = []
    nf = initial_filter_size
    for _ in range(n_levels):
        nf_next = nf*factor
        filters = [nf] + [nf_next]*n_conv_per_level
        nf = nf_next
        down_filters.append(filters)
    
    up_filters = []
    for _ in range(n_levels):
        nf_next = nf//factor
        filters = [nf + nf_next] + [nf_next]*n_conv_per_level
        nf = nf_next
        up_filters.append(filters)
    
    
    InitialBlock = 
    DownBlock,
    UpBlock,
    OutputBlock
    
    
    
    return down_filters, up_filters
        
    
    
if __name__ == '__main__':
    down_filters, up_filters =  unet_progressive(1, 1)
    ConvBlock(1,1)
    
    
    #%%
    

    
    
    
    
    