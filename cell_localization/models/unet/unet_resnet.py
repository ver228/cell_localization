#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:44:12 2019

@author: avelinojaver
"""

from .unet_base import UNetBase, ConvBlock, UpSimple, init_weights


import torch
import math
import collections

from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet
from torchvision.ops.misc import FrozenBatchNorm2d

#dummy variable used for compatibility with the unetbase
NF = collections.namedtuple('NF', 'n_filters')

class FrozenBatchNorm2dv2(FrozenBatchNorm2d):
    def __init__(self, *args, **argkws):
        super().__init__( *args, **argkws)
        #the batchnorm in resnext is missing a variable in the presaved values, but the pytorch does not have it FrozenBatchNorm2d
        #so I am adding it
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

class UnetResnet(UNetBase):
    def __init__(self, 
                 n_inputs,
                 n_outputs, 
                 backbone_name = 'resnet34',
                 batchnorm = False,
                 init_type = 'xavier',
                 pad_mode = 'constant', 
                 return_feat_maps = False,
                 is_imagenet_normalized = True,
                 load_pretrained = True
                 ):
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        self.is_imagenet_normalized = is_imagenet_normalized
        
        
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        
        if n_inputs != 3:
            self.image_mean = sum(self.image_mean)/3
            self.image_std = sum(self.image_std)/3
        
        
        
        if backbone_name == 'resnet34' or backbone_name == 'resnet18':
            nf_initial = 512
            output_block_size = 64    
        else:
            nf_initial = 2048
            output_block_size = 256    
        
        ## REAL DOWNSTREAM
        return_layers = {'layer1':1, 'layer2':2, 'layer3':3, 'layer4':4}
        
        
        #UPSTREAM
        nf = nf_initial
        up_blocks = []
        for ii in range(len(return_layers) - 1):
            nf_next = int(math.ceil(nf/2))
            filters = [nf + nf_next] + [nf_next]*2
            nf = nf_next
            
            #scale_factor = 4 if ii == (len(return_layers) - 1) else 2
            _up = UpSimple(filters, batchnorm = batchnorm, scale_factor = 2)
            
            up_blocks.append(_up)
        
        #OUTPUT
        output_block = nn.Sequential(
                nn.Upsample(scale_factor = 4, mode = 'bilinear', align_corners = False),
                ConvBlock(output_block_size, 64, kernel_size = 3, batchnorm = batchnorm),
                ConvBlock(64, 64, kernel_size = 3, batchnorm = batchnorm),
            )
        
        
        super().__init__([],
                 [],
                 up_blocks,
                 output_block,
                 return_feat_maps = return_feat_maps
                 )
        
        
        ## initial
        self.adaptor = nn.Sequential(
                            ConvBlock(n_inputs, 64, kernel_size = 7, batchnorm = batchnorm),
                            ConvBlock(64, 3, kernel_size = 3, batchnorm = batchnorm)
                        )
        
        #dummy variable used for compatibility with the UNetBase
        self.down_blocks = [NF([ nf_initial // (2**ii)]) for ii in range(len(return_layers))][::-1]
        self.n_levels = len(self.down_blocks)
        
        #real downstream
        norm_layer = FrozenBatchNorm2dv2 if 'resnext' in backbone_name else FrozenBatchNorm2d
        backbone = resnet.__dict__[backbone_name](pretrained = load_pretrained, norm_layer = norm_layer)
        self.backbone = IntermediateLayerGetter(backbone, return_layers)
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        self.loc_head = nn.Conv2d(64, n_outputs, kernel_size = 3, padding = 1)
        
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        #initialize model
        if init_type is not None:
            for part in [self.adaptor, self.up_blocks, self.output_block, self.loc_head]:
                for m in part.modules():
                    init_weights(m, init_type = init_type)
        
    
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
        
        resnet_feats = self.backbone(x)
        
        x_downs = [x for x in resnet_feats.values()]
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
    X = torch.rand((1, 1, 100, 100))
    #feats = backbone(X)
    
    model = UnetResnet(1, 1, return_feat_maps = True, backbone_name = 'resnext101_32x8d')
    xout, feats = model(X)
    
    assert xout.shape[2:] == X.shape[2:]
    
    loss = ((xout - X)**2).mean()
    loss.backward()
    