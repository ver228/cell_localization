#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:13:46 2018

@author: avelinojaver

"""
import torch
from torch import nn

from .unet_base import UNetBase,  UpSimple, DownSimple

def _conv3x3(n_in, n_out):
    return [nn.Conv2d(n_in, n_out, 3, padding=1),
    nn.LeakyReLU(negative_slope=0.1, inplace=True)]


class UNetN2N(UNetBase):
    def __init__(self, 
                 n_inputs,
                 n_outputs,
                 init_type = None,
                 pad_mode = 'constant',
                 **argkws):
        
        initial_block = nn.Sequential(*_conv3x3(n_inputs, 48))
        
        down_blocks = [DownSimple([48, 48]) for _ in range(5)] 
        
        
        up_blocks = [UpSimple([96, 96, 96]),
                     UpSimple([144, 96, 96]),
                     UpSimple([144, 96, 96]),
                     UpSimple([144, 96, 96]),
                     UpSimple([96 + n_inputs, 64, 32])
                     ]
        
        output_block = nn.Sequential(nn.Conv2d(32, n_outputs, 3, padding=1))
        
        super().__init__(
                 initial_block,
                 down_blocks,
                 up_blocks,
                 output_block,
                 init_type = init_type,
                 pad_mode = pad_mode,
                 **argkws)
        
        self.conv6 = nn.Sequential(*_conv3x3(48, 48)) #there is an extra convolution here in the original implementation
        
    def _unet(self, x_input):    
        x = self.initial_block(x_input)
        
        x_downs = [x_input]
        for ii, down in enumerate(self.down_blocks):
            if ii != 0:
                x_downs.append(x)
            x = down(x)
        
        x = self.conv6(x)
        feats = [x]
        
        assert len(x_downs) == len(self.up_blocks)
        for x_down, up in zip(x_downs[::-1], self.up_blocks):
            x = up(x, x_down)
            feats.append(x)
        
        x = self.output_block(x)
        
        return x, feats

if __name__ == '__main__':
    model = UNetN2N(1, 1)
    X = torch.rand((1, 1, 540, 600))
    out = model(X)
    
    print(out.size())
     
    
