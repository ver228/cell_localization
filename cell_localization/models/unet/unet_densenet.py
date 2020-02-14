#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:44:12 2019

@author: avelinojaver
"""

from .unet_base import UNetBase, ConvBlock, init_weights


import torch
import math
import collections

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import densenet
import torch.nn.functional as F

#dummy variable used for compatibility with the unetbase
NF = collections.namedtuple('NF', 'n_filters')
#%%


class UpTransition(nn.Module):
    def __init__(self, 
                    num_inputs,
                    num_outputs,
                    block = None
                ):
        super().__init__()
        
        self.conv = nn.Sequential(
                ConvBlock(num_inputs, num_inputs, 3, batchnorm = True),
                ConvBlock(num_inputs, num_outputs, 3, batchnorm = True)
                )
        
        self.block = None
        
        
        
    def forward(self, x1, x2):
        if self.block is not None:
            x1  = self.block(x1)
        
        x1 = F.interpolate(x1, 
                           x2.shape[-2:],
                           mode = 'bilinear',
                           align_corners=False
                           )
        
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
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
        if n_inputs != 3:
            self.image_mean = sum(self.image_mean)/3
            self.image_std = sum(self.image_std)/3
            
        self.backbone_name = backbone_name
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        super().__init__([], [], [], [], return_feat_maps = return_feat_maps) #dummy initizializaton...
        
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
        
        
        
        base_model = densenet.__dict__[backbone_name]()
        self._up = []
        
        filters_ = self.blocks_num_filters
        for i in [4, 3, 2, 1]:
            dense_block = getattr(base_model.features, f'denseblock{i}')
           
            if i == 4:
                transition_block = ConvBlock(filters_[4], 
                                               filters_[3]//2, 
                                               kernel_size = 1, 
                                               batchnorm = True
                                               )
                up_block = nn.Sequential(transition_block, dense_block)
            else:
                up_block = UpTransition(filters_[i] + filters_[i + 1], 
                                               filters_[i],
                                               block = dense_block
                                               )
            
            
            self.up_blocks.append(up_block)
        
        output_block_size = self.blocks_num_filters[1]
        #OUTPUT
        self.output_block = nn.Sequential(
                nn.Upsample(scale_factor = 4, mode = 'bilinear', align_corners = False),
                ConvBlock(output_block_size, 64, kernel_size = 3, batchnorm = batchnorm),
                ConvBlock(64, 64, kernel_size = 3, batchnorm = batchnorm),
            )
        
        
        #CLASSIFICATION
        self.loc_head = nn.Conv2d(64, n_outputs, kernel_size = 3, padding = 1)
        
        
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
                num_features = num_features // 2
                
            self._blocks_num_filters = n_filters
            return self._blocks_num_filters
    
    def normalize(self, image):
        #taken from https://github.com/pytorch/vision/blob/master/torchvision/models/detection/transform.py
        if self.n_inputs == 3:
            dtype, device = image.dtype, image.device
            mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
            std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
            return (image - self.image_mean) / self.image_std
    
    def _unet(self, x_input):
        if self.is_imagenet_normalized:
            x_input = self.normalize(x_input)
        
        
        x = self.adaptor(x_input)
        
        
        down_feats = [x for x in self.backbone(x).values()][::-1]
        
        
        x = self.up_blocks[0](down_feats[0])
        
        up_feats = [x]
        for x_down, up_block in zip(down_feats[1:], self.up_blocks[1:]):
            x = up_block(x, x_down)
            up_feats.append(x)
        
        x = self.output_block(x)
        up_feats.append(x)
        
        xout = self.loc_head(x)
        
        return xout, up_feats

if __name__ == '__main__':
    
    
    
    X = torch.rand((4, 1, 100, 100))
    #feats = backbone(X)
    
    model = UnetDensenet(1, 1, return_feat_maps = True, backbone_name = 'densenet121')
    
    xout, feats = model(X)
    
    assert xout.shape[2:] == X.shape[2:]
    
    loss = ((xout - X)**2).mean()
    loss.backward()
    #%%
    