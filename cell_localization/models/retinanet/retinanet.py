#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:15:42 2018

@author: avelinojaver
"""
import math
import torch
from torch import nn
from .fpn import FPN, FPNSmall
#%%
class RetinaHead(nn.Module):
    def __init__(self, out_planes, num_anchors, is_classification = False):
        super().__init__()
        self.out_planes = out_planes
        self.num_anchors = num_anchors
        
        layers = []
        for _ in range(4):
            conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self._norm_init_weights(conv)
            layers.append(nn.ReLU())
        
        conv = nn.Conv2d(256, out_planes*num_anchors, kernel_size=3, stride=1, padding=1)
        
        if not is_classification:
            #regression, normal initialization without ReLU
            self._norm_init_weights(conv)
            
            layers.append(conv)
            #layers.append(nn.Tanh())
        else:
            #add bias to make easier to train the classification layer 
            #"every anchor should be labeled as foreground with confidence of ∼π"
            #probability = 0.01
            probability = 0.05
            bias = -math.log((1-probability)/probability)
            nn.init.constant_(conv.bias.data, bias)
            nn.init.constant_(conv.weight.data, 0.0)
            
            layers.append(conv)
            layers.append(nn.Sigmoid())
            
        self._head = nn.Sequential(*layers)
        
        
           
    def _norm_init_weights(self, conv_layer):
        nn.init.normal_(conv_layer.weight.data, mean=0.0, std=0.01)
        nn.init.constant_(conv_layer.bias.data, 0.0)
        return conv_layer
        
    def forward(self, x):
        
        pred = self._head(x)
        batch_size = pred.shape[0]
        pred = pred.permute(0,2,3,1).contiguous().view(batch_size, -1, self.out_planes)
        
        return pred
    
class RetinaNet(nn.Module):
    def __init__(self,  
                 backbone = 'resnet34', 
                 is_fpn_small = False,
                 num_classes=20, 
                 num_anchors = 3
                 ):
        super().__init__()
        
        if is_fpn_small:
            self.fpn = FPNSmall(backbone)
        else:
            self.fpn = FPN(backbone)
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.loc_head = RetinaHead(4, self.num_anchors)
        self.cls_head = RetinaHead(self.num_classes, self.num_anchors, is_classification = True)
        
    def forward(self, x):
        
        if x.shape[1] == 1:
            #add the third dimension for the pretrained resnet
            x = x.repeat(1, 3, 1, 1)
        
        
        fms = self.fpn(x)
        
        loc_preds = [self.loc_head(fm) for fm in fms]
        loc_preds = torch.cat(loc_preds,1)
        
        cls_preds = [self.cls_head(fm) for fm in fms]
        cls_preds = torch.cat(cls_preds,1)
        
        return cls_preds, loc_preds
        
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
#%%
if __name__ == '__main__':
    #%% 
    net = RetinaNet(is_fpn_small = True)
    loc_preds, cls_preds = net(torch.randn(2, 1, 96, 96))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = torch.randn(loc_preds.size())
    cls_grads = torch.randn(cls_preds.size())
    #loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)