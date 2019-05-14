#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))


from cell_localization.models import UNet
import torch
import numpy as np
import tables
import tqdm
import cv2
import matplotlib.pylab as plt





if __name__ == '__main__':
    save_dir =  Path('/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/results/')
    
    bn = 'cell-demixer_unet_l1smooth_20190326_145030_adam_lr0.00032_wd0.0_batch64'
    
    
    #n_epoch = 399 
    #check_name = f'checkpoint-{n_epoch}.pth.tar'
    check_name = 'checkpoint.pth.tar'
    
    model_path = Path('/Volumes/loco/') / 'workspace/localization/results/cell_demixer' / bn / check_name
   
    scale_int = (0, 4095)
    
    n_ch_in, n_ch_out  = 1, 2
    if '-separated' in bn:
        n_ch_out = 3
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #input_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/raw/')
    #fnames = input_dir.glob('*.tif')
    
    input_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/raw')
    fnames = input_dir.glob('*.tif')
    
    for fname in tqdm.tqdm(fnames):
        img = cv2.imread(str(fname), -1)
        
        x = img[None].astype(np.float32)
        x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
        
        with torch.no_grad():
            X = torch.from_numpy(x[None])
            Xhat = model(X)
        
        xhat = Xhat.squeeze().detach().numpy()
        xr = x.squeeze()
        
        #%%
        
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (15, 5))
        axs[0].imshow(xr, cmap='gray', vmin=0, vmax=1)
        
        axs[1].imshow(xhat[0], cmap='gray', vmin=0, vmax=1)
        axs[2].imshow(xhat[1], cmap='gray', vmin=0, vmax=1)

        
        for ax in axs:
            ax.axis('off')
        
        
        #%%
        
        
        #%%
        #fig, ax = plt.subplots(1,1,sharex=True, sharey=True)
        #ax.imshow(xr, cmap='gray')#, vmin=0, vmax=1)
        #ax.imshow(xhat, cmap='inferno', alpha=0.5)
    
    #%%
    
    
    
    
        
    
    