#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:13:46 2018

@author: avelinojaver

"""

from .squeeze_excitation import ChannelSpatialSELayer

import math
import torch.nn.functional as F
import torch
from torch import nn
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

    #print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
    
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

def crop_to_match(x, x_to_crop):
    c = (x_to_crop.size()[2] - x.size()[2])/2
    c1, c2 =  math.ceil(c), math.floor(c)
    
    c = (x_to_crop.size()[3] - x.size()[3])/2
    c3, c4 =  math.ceil(c), math.floor(c)
    
    cropped = F.pad(x_to_crop, (-c3, -c4, -c1, -c2)) #negative padding is the same as cropping
    
    return cropped




def conv_block(n_in, n_out, kernel_size = 3, batchnorm = False, padding = 1):
    conv = nn.Conv2d(n_in, n_out, kernel_size = kernel_size, padding = padding)
    act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    
    if batchnorm:
        return [
                conv,
                nn.BatchNorm2d(n_out), 
                act]
    else:
        return [conv, act]

class Down(nn.Module):
    def __init__(self, 
                 n_filters,
                 downsampling_mode = 'maxpool', 
                 n_conv3x3 = 1, 
                 add_scSE = False,
                 **argkws):
        super().__init__()
        if downsampling_mode == 'maxpool':
            downsampling_l = nn.MaxPool2d(2)
        elif downsampling_mode == 'conv':
            downsampling_l = nn.Conv2d(n_filters[-1], 
                                       n_filters[-1], 
                                       3, 
                                       stride = 2, 
                                       padding = 1
                                       )
        else:
            ValueError('Downsampling mode `{}` not implemented.'.format(downsampling_mode))
        
        _layers = []
        for ii in range(len(n_filters) - 1):
            n_in, n_out = n_filters[ii], n_filters[ii+1]
            _layers += conv_block(n_in, n_out, **argkws)
        
        if add_scSE:
            _layers.append(ChannelSpatialSELayer(n_filters[-1]))
            
        _layers.append(downsampling_l)
        
        self.conv_pooled = nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv_pooled(x)
        return x


class Up(nn.Module):
    def __init__(self, 
                 n_filters, 
                 interp_mode = 'nearest', 
                 add_scSE = False,
                 **argkws):
        super().__init__()
        #self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        
        self.interp_mode = interp_mode
        _layers = []
        for ii in range(len(n_filters) - 1):
            n_in, n_out = n_filters[ii], n_filters[ii+1]
            _layers += conv_block(n_in, n_out, **argkws)
        
        if add_scSE:
            _layers.append(ChannelSpatialSELayer(n_filters[-1]))
        
        self.conv = nn.Sequential(*_layers)

    def forward(self, x1, x2):
        #self.up(x1)
        x1 = F.interpolate(x1, 
                           scale_factor = 2,
                           mode = self.interp_mode
                           )
        x2 = crop_to_match(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, 
                 n_channels = 1, 
                 n_classes = 1, 
                 batchnorm = False,
                 tanh_head = False,
                 init_type = 'xavier',
                 valid_padding = False,
                 interp_mode = 'nearest',
                 downsampling_mode = 'maxpool',
                 pad_mode = 'constant'
                 ):
        super().__init__()
        
        padding = 0 if valid_padding else 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.batchnorm = batchnorm
        self.tanh_head = tanh_head
        self.pad_mode = pad_mode
        
        self.conv0 = nn.Sequential(*conv_block(n_channels, 48, batchnorm = batchnorm, padding = padding))
        
        
        _down_func = partial(Down, 
                        batchnorm = batchnorm,
                        padding = padding,
                        downsampling_mode = downsampling_mode)
        
        self.down1 = _down_func([48, 48])
        self.down2 = _down_func([48, 48])
        self.down3 = _down_func([48, 48])
        self.down4 = _down_func([48, 48])
        self.down5 = _down_func([48, 48])
        
        
        dd = conv_block(48, 48, batchnorm = batchnorm, padding = padding)
        self.conv6 = nn.Sequential(*dd)
        
        _up_func = partial(Up, 
                           interp_mode = interp_mode,
                            batchnorm = batchnorm,
                            padding = padding)
        
        self.up5 = _up_func([96, 96, 96])
        self.up4 = _up_func([144, 96, 96])
        self.up3 = _up_func([144, 96, 96])
        self.up2 = _up_func([144, 96, 96])
        self.up1 = _up_func([96 + n_channels, 64, 32])
        
        
        _layers = []
        if self.tanh_head:
            _layers.append(nn.Tanh)
        _layers.append(nn.Conv2d(32, n_classes, 3, padding = padding))
        self.conv_out = nn.Sequential(*_layers)
        
        #print(f'initialize network with `{init_type}`')
        
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
    
    def _unet(self, x_input):    
        x0 = self.conv0(x_input)
        
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.conv6(x5)
        
        x = self.up5(x6, x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x_input)
        
        x = self.conv_out(x)
        
        return x
    
    
    
    def forward(self, x_input):
        pad_, pad_inv_ = calculate_padding(x_input.shape[2:])
        
        x_input = F.pad(x_input, pad_, mode = self.pad_mode)
        x = self._unet(x_input)
        x = F.pad(x, pad_inv_)
        
        return x
        

class UNetFlatter(nn.Module):
    def __init__(self, 
                 n_channels = 1, 
                 n_classes = 1, 
                 batchnorm = False,
                 out_sigmoid = False,
                 init_type = 'xavier',
                 valid_padding = False,
                 interp_mode = 'nearest',
                 downsampling_mode = 'maxpool',
                 pad_mode = 'constant'
                 ):
        super().__init__()
        
        
        self.valid_padding = valid_padding
        padding = 0 if valid_padding else 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.batchnorm = batchnorm
        self.out_sigmoid = out_sigmoid
        self.pad_mode = pad_mode
        
        self.conv0 = nn.Sequential(*conv_block(n_channels, 48, batchnorm = batchnorm))
        
        
        _down_func = partial(Down, 
                            batchnorm = batchnorm,
                            padding = padding,
                            downsampling_mode = downsampling_mode)
            
        self.down1 = _down_func([48, 48, 48])
        self.down2 = _down_func([48, 48, 48])
        self.down3 = _down_func([48, 48, 48])
        
        
        dd = conv_block(48, 48, batchnorm = batchnorm, padding = padding)
        self.conv4 = nn.Sequential(*dd)
        
        _up_func = partial(Up, 
                           interp_mode = interp_mode,
                            batchnorm = batchnorm,
                            padding = padding)
        
        self.up3 = _up_func([96, 96, 96, 96])
        self.up2 = _up_func([144, 96, 96, 96])
        self.up1 = _up_func([96 + n_channels, 64, 64, 32])
        
        
        dd = nn.Conv2d(32, n_classes, 3, padding = padding)
        self.conv_out = nn.Sequential(dd)
        
        #print(f'initialize network with `{init_type}`')
        
        if init_type is not None:
            for m in self.modules():
                init_weights(m, init_type = init_type)
    
    def _unet(self, x_input):    
        x0 = self.conv0(x_input)
        
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        x4 = self.conv4(x3)
        
        x = self.up3(x4, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x_input)
        
        x = self.conv_out(x)
        
        if self.out_sigmoid:
            x = torch.sigmoid(x)
        
        return x
    
    def forward(self, x_input):
        
        pad_, pad_inv_ = calculate_padding(x_input.shape[2:])
        
        x_input = F.pad(x_input, pad_, mode = self.pad_mode)
        x = self._unet(x_input)
        x = F.pad(x, pad_inv_)
        
        
        return x


class UNetv2(nn.Module):
    def __init__(self, 
                 n_channels = 1, 
                 n_classes = 1, 
                 batchnorm = True,
                 out_sigmoid = False,
                 init_type = 'xavier',
                 valid_padding = False,
                 interp_mode = 'nearest',
                 downsampling_mode = 'maxpool',
                 pad_mode = 'constant',
                 add_scSE = False
                 ):
        super().__init__()
        
        padding = 0 if valid_padding else 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.batchnorm = batchnorm
        self.out_sigmoid = out_sigmoid
        self.pad_mode = pad_mode
        
        _pad = 0 if valid_padding else 3
        _layers = conv_block(n_channels,
                               64, 
                               kernel_size = 7, 
                               batchnorm = batchnorm, 
                               padding = _pad)
        self.conv0 = nn.Sequential(*_layers)
        
        
        _down_func = partial(Down, 
                        batchnorm = batchnorm,
                        padding = padding,
                        downsampling_mode = downsampling_mode,
                        add_scSE = add_scSE)
        
        self.down1 = _down_func([64, 128, 128])
        self.down2 = _down_func([128, 256, 256])
        self.down3 = _down_func([256, 512, 512])
        self.down4 = _down_func([512, 512, 512])
        
        
        dd = conv_block(48, 48, batchnorm = batchnorm, padding = padding) #THIS SHOULDN'T BE HERE BUT IT WILL CRASH OLD MODELS...
        self.conv5 = nn.Sequential(*dd)
        
        _up_func = partial(Up, 
                           interp_mode = interp_mode,
                            batchnorm = batchnorm,
                            padding = padding,
                            add_scSE = add_scSE)
        
        self.up4 = _up_func([1024, 256, 256])
        self.up3 = _up_func([512, 128]) #IT SEEMS THAT I FORGOT TO ADD THE EXTRA CONV2 HERE...
        self.up2 = _up_func([256, 64])
        self.up1 = _up_func([128, 32])
        
        
        dd = nn.Conv2d(32, n_classes, 3, padding = padding)
        self.conv_out = nn.Sequential(dd)
        
        #print(f'initialize network with `{init_type}`')
        
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
        
        if self.out_sigmoid:
            x = torch.sigmoid(x)
        
        return x
    
    
    
    def forward(self, x_input):
        pad_, pad_inv_ = calculate_padding(x_input.shape[2:])
        
        x_input = F.pad(x_input, pad_, mode = self.pad_mode)
        x = self._unet(x_input)
        x = F.pad(x, pad_inv_)
        
        return x

class UNetv2B(nn.Module):
    def __init__(self, 
                 n_channels = 1, 
                 n_classes = 1, 
                 batchnorm = True,
                 tanh_head = False,
                 sigma_out = False,
                 init_type = 'xavier',
                 valid_padding = False,
                 interp_mode = 'nearest',
                 downsampling_mode = 'maxpool',
                 pad_mode = 'constant',
                 add_scSE = False
                 ):
        super().__init__()
        
        padding = 0 if valid_padding else 1
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.batchnorm = batchnorm
        self.tanh_head = tanh_head
        self.sigma_out = sigma_out
        self.pad_mode = pad_mode
        
        _pad = 0 if valid_padding else 3
        _layers = conv_block(n_channels,
                               48, 
                               kernel_size = 7, 
                               batchnorm = batchnorm, 
                               padding = _pad)
        self.conv0 = nn.Sequential(*_layers)
        
        
        _down_func = partial(Down, 
                        batchnorm = batchnorm,
                        padding = padding,
                        downsampling_mode = downsampling_mode,
                        add_scSE = add_scSE)
        
        self.down1 = _down_func([48, 96, 96])
        self.down2 = _down_func([96, 192, 192])
        self.down3 = _down_func([192, 384, 384])
        self.down4 = _down_func([384, 768, 768])
        
        _up_func = partial(Up, 
                           interp_mode = interp_mode,
                            batchnorm = batchnorm,
                            padding = padding,
                            add_scSE = add_scSE)
        
        self.up4 = _up_func([1152, 384, 384])
        self.up3 = _up_func([576, 192, 192])
        self.up2 = _up_func([288, 96, 96])
        self.up1 = _up_func([144, 48, 48])
        
        _layers = []
        if self.tanh_head:
            _layers.append(nn.Tanh())
        _layers.append(nn.Conv2d(48, n_classes, 3, padding = padding))
        if self.sigma_out:
            _layers.append(nn.Sigmoid())
            
        self.conv_out = nn.Sequential(*_layers)
        
        #print(f'initialize network with `{init_type}`')
        
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
#%%
class UNetscSE(UNetv2):
    def __init__(self, *args, **argkws):
        super().__init__(*args, add_scSE = True, **argkws)
    
#%%
if __name__ == '__main__':
    n_in = 3
    n_out = 2
    batchnorm = True
    sigma_out = True
    valid_padding = False
    interp_mode = 'nearest'
    downsampling_mode = 'maxpool'
    
    mod = UNetv2B(n_in, 
               n_out, 
               batchnorm = batchnorm,
               sigma_out = sigma_out,
               valid_padding = valid_padding,
               interp_mode = interp_mode,
               downsampling_mode = downsampling_mode
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