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

from cell_localization.flow import CoordFlow, collate_pandas
from cell_localization.models.losses import LossWithBeliveMaps
     

import tqdm
from torch.utils.data import DataLoader
#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')


import matplotlib.pylab as plt

def main():
    #%%
    #root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/rois/20x/train'
    #root_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/20x/train'
    #root_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/no_membrane/train'
    #root_dir = Path.home() / 'workspace/localization/data/heba/data-uncorrected/train'   
    
    #root_dir = Path.home() / 'OneDrive - Nexus365/heba/WoundHealing/data4train/mix/train'
    root_dir =  Path.home() / 'workspace/localization/data/woundhealing/annotated/v2/mix/validation'
    
    #root_dir = Path.home() / 'workspace/localization/test_images/'
    #loc_gauss_sigma = 2
    
#    root_dir = Path.home() / 'workspace/localization/data/woundhealing/demixed_predictions'
#    loc_gauss_sigma = 2
#    roi_size = 48
    
    #root_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam/validation'
    #root_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam_refined/validation'
    
    num_workers = 4
    batch_size = 512
    gauss_sigma = 1.5
    
    flow_args = dict(
                roi_size = 48,
                scale_int = (0, 4095),
                prob_unseeded_patch = 0.0,
              
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
                        collate_fn = collate_pandas
                        )
    
    
    maps_getter = LossWithBeliveMaps(gauss_sigma = gauss_sigma)
    for X, targets in tqdm.tqdm(loader):
        belive_map = maps_getter.targets2belivemaps(targets, X.shape)
#        for x, maps, target in zip(X, belive_map, targets):
#            fig, axs = plt.subplots(1, 1 + maps.shape[0])
#            
#            xr = x[0].detach().numpy()
#            
#            for ax in axs:
#                ax.imshow(xr, cmap = 'gray')
#            
#            coords = target['coordinates'].detach().numpy()
#            labels = target['labels'].detach().numpy()
#            for i, m in enumerate(maps):
#                m = m.detach().numpy()
#                axs[i+1].imshow(m, cmap='inferno', alpha=0.5)
#                
#                good = labels == i+1
#                axs[i+1].plot(coords[good, 0], coords[good, 1], '.r')
           
        
if __name__ == '__main__':
    import fire
    fire.Fire(main)        