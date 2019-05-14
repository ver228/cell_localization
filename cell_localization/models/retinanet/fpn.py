#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:11:29 2018

@author: avelinojaver
"""

import os
from pathlib import Path

pretrained_path = str(Path.home() / 'workspace/pytorch/pretrained_models/')
os.environ['TORCH_MODEL_ZOO'] = pretrained_path


import torch
from torch import nn
import torch.nn.functional as F

import torchvision

_resnet = {
        'resnet18' : torchvision.models.resnet18,
        'resnet34' : torchvision.models.resnet34,
        'resnet50' : torchvision.models.resnet50,
        'resnet101' : torchvision.models.resnet101,
        }




class FPN(nn.Module):
    '''
    Feature Pyramid Network
    '''
    def __init__(self, backbone = 'resnet50'):
        super().__init__()
        
        #right now I only will support the models in the pytorch zoo maybe later I can improve this...
        assert backbone in _resnet
        model_base = _resnet[backbone](pretrained=True)
        
        self.layer0  = nn.Sequential(
                        model_base.conv1,
                        model_base.bn1,
                        model_base.relu,
                        model_base.maxpool)
        
        self.layer1  = model_base.layer1
        self.layer2  = model_base.layer2
        self.layer3  = model_base.layer3
        self.layer4  = model_base.layer4
        
        if backbone in ['resnet18', 'resnet34']:
            bn_layer = 'conv2'
        else:
            bn_layer = 'conv3'
        
        c5_out = getattr(self.layer4[-1], bn_layer).out_channels
        c4_out = getattr(self.layer3[-1], bn_layer).out_channels
        c3_out = getattr(self.layer2[-1], bn_layer).out_channels
        
        
        self.conv6 = nn.Conv2d(c5_out, 256, kernel_size=3, stride=2, padding=1)
        
        self.conv7 = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
                        )

        # Lateral layers
        self.latlayer1 = nn.Conv2d(c5_out, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(c4_out, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(c3_out, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        p6 = self.conv6(c5)
        p7 = self.conv7(p6)
        
        
        
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        
        return p3, p4, p5, p6, p7
        
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, 
                             size=(H,W), 
                             mode='bilinear',
                             align_corners = False) + y
                             
                             
class FPNSmall(nn.Module):
    '''
    Feature Pyramid Network
    '''
    def __init__(self, backbone = 'resnet50'):
        super().__init__()
        
        #right now I only will support the models in the pytorch zoo maybe later I can improve this...
        assert backbone in _resnet
        model_base = _resnet[backbone](pretrained=True)
        
        self.layer0  = nn.Sequential(
                        model_base.conv1,
                        model_base.bn1,
                        model_base.relu,
                        )
        
        self.layer1  = nn.Sequential(
                            model_base.maxpool,
                            model_base.layer1)
        self.layer2  = model_base.layer2
        self.layer3  = model_base.layer3
        self.layer4  = model_base.layer4
        
        if backbone in ['resnet18', 'resnet34']:
            bn_layer = 'conv2'
        else:
            bn_layer = 'conv3'
        
        c5_out = getattr(self.layer4[-1], bn_layer).out_channels
        c4_out = getattr(self.layer3[-1], bn_layer).out_channels
        c3_out = getattr(self.layer2[-1], bn_layer).out_channels
        c2_out = getattr(self.layer1[1][-1], bn_layer).out_channels
        c1_out = self.layer0[0].out_channels
        
        # Lateral layers
        self.latlayer1 = nn.Conv2d(c5_out, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(c4_out, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(c3_out, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(c2_out, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer5 = nn.Conv2d(c1_out, 256, kernel_size=1, stride=1, padding=0)
        
        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        c1 = self.layer0(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        #p6 = self.conv6(c5)
        #p7 = self.conv7(p6)
        
        
        
        p5 = self.latlayer1(c5)
        
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)
        
        p2 = self._upsample_add(p3, self.latlayer4(c2))
        p2 = self.toplayer3(p2)
        
        p1 = self._upsample_add(p2, self.latlayer5(c1))
        p1 = self.toplayer4(p1)
        
        
        return p1, p2, p3, p4, p5
    
    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        x_up = F.interpolate(x, 
                             size=(H,W), 
                             mode='bilinear',
                             align_corners = False)
        return x_up + y
#%%
if __name__ == '__main__':
    
    
    mod = FPNSmall('resnet18')
    
    X = torch.zeros(1, 3, 128, 128)
    feats = mod(X)
    
    print([x.shape for x in feats])
    