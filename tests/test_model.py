#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:01:35 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

import torch
from torch import nn


from cell_localization.models.cell_detector_with_clf import CellDetectorWithClassifier, SimpleClassifier


        
if __name__ == '__main__':
    
    n_in = 3
    n_out = 2
    batchnorm = True
    im_size  = 128, 128
    X = torch.rand((1, n_in, *im_size))
    target = torch.rand((1, n_out, *im_size))
    
    model = CellDetectorWithClassifier(unet_n_inputs = n_in, unet_n_ouputs = n_out)
    xhat, features = model.mapping_network(X)
    #%%
    
    
    
    n_filters_clf = model.mapping_network.down_blocks[-1].n_filters[-1]
    clf_head = SimpleClassifier(n_filters_clf, 2)
   
    dd = clf_head(features[0])