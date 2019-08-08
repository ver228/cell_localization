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

from cell_localization.flow import CoordFlow, collate_simple
from cell_localization.models.cell_detector_with_clf import CellDetectorWithClassifier
     
import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np

import matplotlib.pylab as plt

if __name__ == '__main__':

    #%%
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/eosinophils/training/20x/'
    
    num_workers = 4
    batch_size = 128#512
    gauss_sigma = 1.5
    device = 'cpu'
    
    flow_args = dict(
                roi_size = 96,
                scale_int = (0, 4095),
                prob_unseeded_patch = 0.5,
              
                zoom_range = (0.97, 1.03),
                
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.7, 1.3),
                samples_per_epoch = batch_size*100
                )
    
    gen = CoordFlow(root_dir, **flow_args)
    
    loader = DataLoader(gen,
                        batch_size = batch_size, 
                        shuffle = True, 
                        num_workers = num_workers,
                        collate_fn = collate_simple
                        )
    
    model = CellDetectorWithClassifier(unet_n_inputs = 3, unet_n_ouputs = 2)
    
    for images, targets in tqdm.tqdm(loader):
        images = torch.from_numpy(np.stack(images)).to(device)
        targets = [{k: torch.from_numpy(v).to(device) for k, v in target.items()} for target in targets]
        
        loss = model(images, targets)
        break
