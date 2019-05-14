#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:55:24 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))

import torch
import pandas as pd

import numpy as np
import matplotlib.pylab as plt
import tqdm
from cell_localization.models import UNet

#from from_images import cv2_peak_local_max
from skimage.feature import peak_local_max
from cell_localization.evaluation.localmaxima import evaluate_coordinates

if __name__ == '__main__':
    bn = 'eggs-int/eggs-int_unet_hard-neg-freq1_l1smooth_20190513_081902_adam_lr0.000128_batch128'
    #bn = 'eggs-int/eggs-int_unet_l1smooth_20190512_210150_adam_lr0.000128_batch128'
    #bn = 'eggs-only/eggs-only_unet-bn-sigmoid_hard-neg-freq1_l1smooth_20190514_094134_adam_lr0.0008_batch8'
    #bn = 'eggs-only/eggs-only_unet-bn_hard-neg-freq1_l1smooth_20190514_131259_adam_lr8e-05_batch8'
    
    n_epoch = None
    
    if n_epoch is None:
        #check_name = 'checkpoint.pth.tar'
        check_name = 'model_best.pth.tar'
    else:
        check_name = f'checkpoint-{n_epoch}.pth.tar'
    
    model_path = Path().home() / 'workspace/localization/results/locmax_detection' / bn / check_name
    print(model_path)
    
    
    n_ch_in, n_ch_out  = 1, 1
    batchnorm = 'unet-bn' in bn
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm=batchnorm)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    print(state['epoch'])
    #%%
#    data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs/validation'
#    
#    #data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs/train'
#    
#    data_root_dir = Path(data_root_dir)
#    fnames = data_root_dir.rglob('*.hdf5')
#    fnames = list(fnames)
    
    fnames = []
    for dd in ['validation', 'test']:
        data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs/' / dd
        fnames += list(data_root_dir.rglob('*.hdf5'))
    #%%
    norm_args = dict(vmin = 0, vmax=1)
    
    metrics = np.full((model.n_classes, 3), 1e-3)
    for fname in tqdm.tqdm(fnames):
        with pd.HDFStore(str(fname), 'r') as fid:
            img = fid.get_node('/img')[:]
            df = fid['/coords']
        
        #xin = np.rollaxis(img, 2, 0)
        xin = img[None]
        xin = xin.astype(np.float32)/255
        
        with torch.no_grad():
            xin = torch.from_numpy(xin[None])
            xhat = model(xin)
        
        xout = xhat[0].detach().numpy()
        #%%
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        
        axs[0].imshow(img, cmap='gray')
        axs[1].imshow(xout[0], **norm_args)
        
        coords_pred = peak_local_max(xout[0], min_distance = 2, threshold_abs = 0.05, threshold_rel = 0.1)
        target = np.array((df['cy'], df['cx'])).T
       
        TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(coords_pred, target, max_dist = 5)
        metrics[0] += (TP, FP, FN)
        
        axs[0].plot(df['cx'], df['cy'], 'xr')
        if coords_pred.size > 0:
            axs[0].plot(coords_pred[...,1], coords_pred[...,0], 'g.')
        #%%
        
    plt.show()
    
    
    TP, FP, FN = metrics[0]
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    
    
    print(f'P={P}, R={R}, F1={F1}')
    
    #%%
    print(bn)