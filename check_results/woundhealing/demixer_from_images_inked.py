#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from cell_localization.models import UNet
import torch
import numpy as np
import tqdm
import cv2
import matplotlib.pylab as plt





if __name__ == '__main__':
    
    #bn = 'inkedslidesv2-roi128_unet_l1smooth_20190605_230147_adam_lr6.4e-05_wd0.0_batch64'
    #bn = 'inkedslides-roi256_unet_l1smooth_20190605_213434_adam_lr6.4e-05_wd0.0_batch64'
    #bn = 'inkedslides-roi128_unet_l1smooth_20190605_213037_adam_lr6.4e-05_wd0.0_batch64'
    #bn = 'inkedslidesv2-roi256_unet_l1smooth_20190606_082516_adam_lr6.4e-05_wd0.0_batch64'
    #bn = 'inkedslidesv3-roi512_unet_l1smooth_20190606_155659_adam_lr6e-05_wd0.0_batch12'
    bn = 'inkedslidesv3-roi256_unet_l1smooth_20190606_155655_adam_lr0.00016_wd0.0_batch32'
    
    check_name = 'checkpoint.pth.tar'
    
    model_path = Path.home() / 'workspace/localization/results/cell_demixer' / bn.partition('-')[0] / bn /check_name
    
    scale_int = (0, 255)
    
    n_ch_in, n_ch_out  = 3, 6
    if '-separated' in bn:
        n_ch_out = 3
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out, pad_mode = 'reflect')
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    
    #input_dir = Path.home() / 'workspace/denoising/data/inked_slides/test_ISBI/'
    input_dir = Path.home() / 'workspace/denoising/data/inked_slides/samples/'
    
    fnames = input_dir.glob('*.jpg')
    
    fnames =  [x for x in fnames if not x.name.startswith('.')]
    for fname in tqdm.tqdm(fnames):
        img = cv2.imread(str(fname), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        x = img.astype(np.float32)
        x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
        x = np.rollaxis(x, 2, 0)
        
        with torch.no_grad():
            X = torch.from_numpy(x[None])
            Xhat = model(X)
        #%%
        xhat = Xhat[0].detach().numpy()
        xhat = np.rollaxis(xhat, 0, 3)
        
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (15, 5))
        
        
        axs[0].imshow(img, cmap='gray')#, vmin=0, vmax=1)
        
        
        axs[1].imshow(xhat[..., :3], cmap='gray')#, vmin=0, vmax=1)
        axs[2].imshow(xhat[..., -3:], cmap='gray')#, vmin=0, vmax=1)

        
        for ax in axs:
            ax.axis('off')
        
        plt.suptitle(bn)
        
        #%%
        
        
        
    
    