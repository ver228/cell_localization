#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 14:04:43 2019

@author: avelinojaver
"""
import torch
from torch import nn
import torch.nn.functional as F

import math
from functools import partial

def init_weights(net, init_type='xavier', init_gain=0.02):
    """Initialize network weights.
    Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)


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
        self.n_filters = n_filters
        
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
    
class UpSimple(nn.Module):
    def __init__(self, 
                 n_filters, 
                 interp_mode = 'nearest', 
                 **argkws):
        super().__init__()
        self.n_filters = n_filters
        
        self.interp_mode = interp_mode
        _layers = []
        for ii in range(len(n_filters) - 1):
            n_in, n_out = n_filters[ii], n_filters[ii+1]
            _layers += ConvBlock(n_in, n_out, **argkws)
        self.conv = nn.Sequential(*_layers)

    @staticmethod
    def crop_to_match(x, x_to_crop):
        c = (x_to_crop.size()[2] - x.size()[2])/2
        c1, c2 =  math.ceil(c), math.floor(c)
        
        c = (x_to_crop.size()[3] - x.size()[3])/2
        c3, c4 =  math.ceil(c), math.floor(c)
        
        cropped = F.pad(x_to_crop, (-c3, -c4, -c1, -c2)) #negative padding is the same as cropping
        
        return cropped

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, 
                           scale_factor = 2,
                           mode = self.interp_mode
                           )
        x2 = self.crop_to_match(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        
        return x
    
class UNetBase(nn.Module):
    def __init__(self, 
                 initial_block,
                down_blocks,
                up_blocks,
                output_block,
                 pad_mode = 'constant'
                 ):
        super().__init__()
        
        
        self.initial_block = initial_block
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_block = output_block
        
        #used only if the image size does not match the corresponding unet levels
        self.pad_mode = pad_mode
        
        
    
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
def unet_constructor(n_inputs,
                     n_ouputs, 
                     DownBlock = None,
                     UpBlock = None,
                     InitialBlock = None,
                     OutputBlock = None,
                     initial_filter_size = 48, 
                     levels = 4, 
                     conv_per_level = 2,
                     increase_factor = 2,
                     batchnorm = False,
                     init_type = 'xavier',
                     pad_mode = 'constant'
                     ):
    
    down_filters = []
    nf = initial_filter_size
    for _ in range(levels):
        nf_next = int(nf*increase_factor)
        filters = [nf] + [nf_next]*conv_per_level
        nf = nf_next
        down_filters.append(filters)
    
    up_filters = []
    for _ in range(levels):
        nf_next = int(math.ceil(nf/increase_factor))
        filters = [nf + nf_next] + [nf_next]*conv_per_level
        nf = nf_next
        up_filters.append(filters)
    
    if DownBlock is None:
        DownBlock = DownSimple

    if UpBlock is None:
        UpBlock = UpSimple

    if InitialBlock is None:
        InitialBlock = partial(ConvBlock, kernel_size = 7)

    if OutputBlock is None:
        def OutputBlock(n_in, n_out, **argws):
            return nn.Conv2d(n_in, n_out, kernel_size = 3, padding = 1)
    
    initial_block = InitialBlock(n_inputs, initial_filter_size, batchnorm = batchnorm)
    down_blocks = [DownBlock(x, batchnorm = batchnorm) for x in down_filters]
    up_blocks = [UpBlock(x, batchnorm = batchnorm) for x in up_filters]
    output_block = OutputBlock(initial_filter_size, n_ouputs, batchnorm = batchnorm)
    
    model = UNetBase(initial_block, down_blocks, up_blocks, output_block, pad_mode = pad_mode)

    #initialize model
    if init_type is not None:
        for m in model.modules():
            init_weights(m, init_type = init_type)

    
    return model

if __name__ == '__main__':
    n_in = 3
    n_out = 2
    batchnorm = True
    im_size  = 196, 196
    X = torch.rand((1, n_in, *im_size))
    target = torch.rand((1, n_out, *im_size))
    
    model = unet_constructor(n_in, n_out, DownSimple, UpSimple, batchnorm = batchnorm)
    out = model(X)
    
    loss = (out-target).abs().sum()
    loss.backward()
    
    print(out.size())
    