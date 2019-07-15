#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:03:09 2019

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from cell_localization.trainer import get_device
from cell_localization.models import UNet

import numpy as np
import torch
import cv2
from pathlib import Path


if __name__ == '__main__':
    import matplotlib.pylab as plt
    import tqdm
    
    _debug = True
    
    root_dir = '/Volumes/rescomp1/data/localization/data/cell_detection/raw/'
    root_dir = Path(root_dir)
    
    
    model_root = '/Volumes/rescomp1/data/localization/results/cell_demixer/'
    model_root = Path(model_root)
    model_path = model_root / 'cell-demixer_unet_l1smooth_20190326_145030_adam_lr0.00032_wd0.0_batch64/checkpoint.pth.tar'
    
    
    cuda_id = 0
    
    device = get_device(cuda_id)
    
    n_ch_in, n_ch_out = 1, 2
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    
    state = torch.load(str(model_path), map_location = str(device))
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    fnames = [x for x in root_dir.glob('*.tif') if not any([x.name[0] == c for c in '.MN'])]
    for fname in tqdm.tqdm(fnames):
        img = cv2.imread(str(fname), -1)
        
        with torch.no_grad():
            xin = torch.from_numpy(img.astype(np.float32))
            xin /= 4095.
            xout = model(xin[None, None])
            
        img_nuclei, img_membrane = xout.squeeze(0).detach().cpu().numpy()
        img_nuclei = np.clip(img_nuclei*4095, 0, 4095).astype(np.uint16)
        img_membrane = np.clip(img_membrane*4095, 0, 4095).astype(np.uint16)
        
        
        cv2.imwrite(str(fname.parent / ('M_' + fname.name)), img_membrane)
        cv2.imwrite(str(fname.parent / ('N_' + fname.name)), img_nuclei)
        
        if _debug:
            fig, axs = plt.subplots(3, 1, figsize = (10, 10), sharex=True, sharey=True)
            axs[0].imshow(img, cmap='gray')
            axs[1].imshow(img_nuclei, cmap='gray')
            axs[2].imshow(img_membrane, cmap='gray')
            for ax in axs:
                ax.axis('off')
            
            plt.suptitle(fname.name)
        
        