#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:20:30 2019

@author: avelinojaver
"""


#from torchvision.models import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FasterRCNN

#%%

class FasterRCNNFixedSize(nn.Module):
    def __init__(self, n_classes, box_size = 7, backbone_name = 'resnet50', pretrained_backbone = True):
        super().__init__()
        
        _dum = set(dir(self))
        self.n_classes = n_classes
        self.pretrained_backbone = pretrained_backbone
        self.box_size = box_size
        self.half_box_size = box_size//2
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        
        anchor_sizes = ((4,), (8,), (16,), (32,), (64,))
        aspect_ratios = ((1.0,),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        backbone = resnet_fpn_backbone(backbone_name, pretrained_backbone)
        # i am using n_classes + 1 because 0 correspond to the background in the torchvision fasterrcnn convension
        self.fasterrcnn = FasterRCNN(backbone, n_classes + 1, rpn_anchor_generator = rpn_anchor_generator)
        
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
        
    def forward(self, X, targets = None):
        
        if self.training: # I cannot accept patches without coordinates
            dd = [(x,t) for x, t in zip(X, targets) if t['coordinates'].numel() > 0]
            X, targets = map(list, zip(*dd))
        
        
        if targets is not None:
            #create the boxes as predictions...
            for t in targets:
                coords = t['coordinates'].float()
                t['boxes'] = torch.cat([coords - self.half_box_size, coords + self.half_box_size], axis=-1)
            
        
            
        output = self.fasterrcnn(X, targets)
        
        if not self.training:
            for t in output:
                boxes = t['boxes']
                t['coordinates'] = boxes[:, :2] + (boxes[:, 2:]-boxes[:, :2])/2
        
            if (targets is not None):
                #here i am including an "empty" loss in order to be consistent with the other code.
                #Ideally i would want to know the loss on the validation test but i will need to modify 
                #the original fasterrcnn code to do this. Since it is not very important I will just return an empyt dict
                output = ({}, output)
                
        
        return output
        
        
    


if __name__ == '__main__':
    model = FasterRCNNFixedSize(num_classes = 2)
    
    X = torch.rand((3, 512, 512))
    targets = [{
            'coordinates' : torch.tensor([(10, 100), (300, 400)]).float(),
            'labels' : torch.tensor([1, 1])
            },
    
            {
            'coordinates' : torch.zeros((0, 2)).float(),
            'labels' : torch.zeros(0)
            }
        ]
    
    
    out = model([X, X], targets)
    
    loss = sum([x for x in out.values()])
    loss.backward()
    
    