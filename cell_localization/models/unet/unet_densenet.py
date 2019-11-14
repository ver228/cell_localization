#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:44:12 2019

@author: avelinojaver
"""

from unet_base import UNetBase, ConvBlock, UpSimple, init_weights


import torch
import math
import collections

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import densenet
from torchvision.models.densenet import _DenseBlock
import torch.nn.functional as F

#dummy variable used for compatibility with the unetbase
NF = collections.namedtuple('NF', 'n_filters')
#%%


class UpDenseBlock(nn.Module):
    def __init__(self, 
                    num_inputs,
                    dense_num_layers,
                    dense_num_input,
                    dense_growth_rate,
                    scale_factor = 2,
                    interp_mode = 'nearest'
                ):
        super().__init__()
        
        self.denseblock = _DenseBlock(
                            num_layers=dense_num_layers,
                            num_input_features=dense_num_input,
                            growth_rate = dense_growth_rate,
                            bn_size=4,
                            drop_rate=0)
        self.conv1 = nn.Conv2d(num_inputs, dense_num_input, 1)
        
        self.scale_factor = scale_factor
        
        self.interp_mode = interp_mode
        
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
                           scale_factor = self.scale_factor,
                           mode = self.interp_mode
                           )
        x2 = self.crop_to_match(x1, x2)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.denseblock(x)
        return x
    
#%%

class UnetDensenet(UNetBase):
    densenet_params = {
        'densenet121' : (32, (6, 12, 24, 16), 64),
        'densenet161' : (48, (6, 12, 36, 24), 96),
        'densenet169' : (32, (6, 12, 32, 32), 64),
        'densenet201' : (32, (6, 12, 48, 32), 64)
        }
    
    
    
    def __init__(self, 
                 n_inputs,
                 n_outputs, 
                 backbone_name = 'densenet121',
                 batchnorm = False,
                 init_type = 'xavier',
                 pad_mode = 'constant', 
                 return_feat_maps = False,
                 is_imagenet_normalized = True
                 ):
        self.is_imagenet_normalized = is_imagenet_normalized
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        
        self.backbone_name = backbone_name
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        super().__init__([], [], [], [], return_feat_maps = return_feat_maps)
        
        ## initial
        self.adaptor = nn.Sequential(
                            ConvBlock(n_inputs, 64, kernel_size = 7, batchnorm = batchnorm),
                            ConvBlock(64, 3, kernel_size = 3, batchnorm = batchnorm)
                        )
        
        ## REAL DOWNSTREAM
        return_layers = {'denseblock1':1, 'denseblock2':2, 'denseblock3':3, 'denseblock4':4}
        #real downstream
        backbone = densenet.__dict__[backbone_name](pretrained = True)
        self.backbone = IntermediateLayerGetter(backbone.features, return_layers)
        for name, parameter in backbone.named_parameters():
            if 'denseblock2' not in name and 'denseblock3' not in name and 'denseblock4' not in name:
                parameter.requires_grad_(False)
        
        #UPSTREAM
        growth_rate, block_config, num_init_features = self.densenet_params[backbone_name]
        num_features = num_init_features
        
        up_blocks = []
        
        model.blocks_num_filters[::-1]
        for i, num_layers in enumerate(block_config[-2::-1]):
            num_inputs = model.blocks_num_filters[-(i + 1)] + model.blocks_num_filters[-(i + 2)]
            dense_num_input = model.blocks_num_filters[-(i + 2)]//2
            up_block = UpDenseBlock(
                num_inputs,
                dense_num_layers = num_layers,
                dense_num_input = dense_num_input,
                dense_growth_rate = growth_rate
            )
            up_blocks.append(up_block)
            
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                num_features = num_features // 2
        #up_blocks = up_blocks[::-1]
        self.up_blocks = nn.ModuleList(up_blocks)
        
        
        #FINAL BLOCK
        self.loc_head = nn.Conv2d(64, n_outputs, kernel_size = 3, padding = 1)
        
        
        output_block_size = 128
        #OUTPUT
        self.output_block = nn.Sequential(
                nn.Upsample(scale_factor = 4, mode = 'bilinear', align_corners = False),
                ConvBlock(output_block_size, 64, kernel_size = 3, batchnorm = batchnorm),
                ConvBlock(64, 64, kernel_size = 3, batchnorm = batchnorm),
            )
        
        
        
        
        #dummy variable used for compatibility with the UNetBase
        self.down_blocks = [NF(x) for x in self.blocks_num_filters[1:]]
        self.n_levels = len(self.down_blocks)
        
        
        #initialize model
        if init_type is not None:
            for part in [self.adaptor, self.up_blocks, self.output_block, self.loc_head]:
                for m in part.modules():
                    init_weights(m, init_type = init_type)
    
    @property
    def blocks_num_filters(self):
        try:
            return self._blocks_num_filters
        except:
            growth_rate, block_config, num_init_features = self.densenet_params[self.backbone_name]
            num_features = num_init_features
            
            n_filters = []
            
            n_filters.append(num_features)
            for i, num_layers in enumerate(block_config):
                num_features = num_features + num_layers * growth_rate
                n_filters.append(num_features)
                if i != len(block_config) - 1:
                    num_features = num_features // 2
                    
            self._blocks_num_filters = n_filters
            return self._blocks_num_filters
    
    def normalize(self, image):
        #taken from https://github.com/pytorch/vision/blob/master/torchvision/models/detection/transform.py
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]
    
    def _unet(self, x_input):
        if self.is_imagenet_normalized:
            x_input = self.normalize(x_input)
        
        x = self.adaptor(x_input)
        
        feats = self.backbone(x)
        
        x_downs = [x for x in feats.values()]
        x_downs = x_downs[::-1]
        x, x_downs = x_downs[0], x_downs[1:]
        
        feats = [x]
        for x_down, up in zip(x_downs, self.up_blocks):
            x = up(x, x_down)
            feats.append(x)
        
        x = self.output_block(x)
        feats.append(x)
        
        xout = self.loc_head(x)
        
        return xout, feats

if __name__ == '__main__':
    
    
    
    X = torch.rand((1, 3, 100, 100))
    #feats = backbone(X)
    
    model = UnetDensenet(3, 1, return_feat_maps = True, backbone_name = 'densenet121')
    feats = model.backbone(X)
    assert all([x.shape[1] == m.n_filters for x, m in zip(feats.values(), model.down_blocks)])
    
    xout, feats = model(X)
    
    assert xout.shape[2:] == X.shape[2:]
    
    loss = ((xout - X)**2).mean()
    loss.backward()
    #%%
    backbone_name = 'densenet121'
    growth_rate, block_config, num_init_features = model.densenet_params[backbone_name]
    num_features = num_init_features
    
    blocks = []
    for i, num_layers in enumerate(block_config):
        block = UpDenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            growth_rate = growth_rate,
            bn_size=4,
            drop_rate=0
        )
        blocks.append(block)
        
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            num_features = num_features // 2
    #up_blocks = up_blocks[::-1]