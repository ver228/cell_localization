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

import cv2
import numpy as np
import matplotlib.pylab as plt
import tqdm
from cell_localization.models import UNet, UNetv2B

#from from_images import cv2_peak_local_max
from skimage.feature import peak_local_max
from cell_localization.evaluation.localmaxima import evaluate_coordinates


def normalize_softmax(xhat):
    n_batch, n_channels, w, h = xhat.shape
    hh = xhat.view(n_batch, n_channels, -1)
    hh = torch.nn.functional.softmax(hh, dim = 2)
    hmax, _ = hh.max(dim=2)
    hh = hh/hmax.unsqueeze(2)
    hh = hh.view(n_batch, n_channels, w, h)
    return hh

def reshape_norm(xhat):
    n_batch, n_outs, w, h = xhat.shape
    n_channels = n_outs // 5
    hh = xhat.view(n_batch, n_channels, 5, w, h)
    hh = hh[:, :, 0]
    hh = normalize_softmax(hh)
    return hh

if __name__ == '__main__':
    #bn = 'eggs-int/eggs-int_unet_hard-neg-freq1_l1smooth_20190513_081902_adam_lr0.000128_batch128'
    #bn = 'eggs-int/eggs-int_unet_l1smooth_20190512_210150_adam_lr0.000128_batch128'
    #bn = 'eggs-only/eggs-only_unet-bn-sigmoid_hard-neg-freq1_l1smooth_20190514_094134_adam_lr0.0008_batch8'
    #bn = 'eggs-only/eggs-only_unet-bn_hard-neg-freq1_l1smooth_20190514_131259_adam_lr8e-05_batch8'
    
    #bn = 'eggsadam-roi48/eggsadam-roi48_unetv2b_l1smooth_20190605_110638_adam_lr0.000128_wd0.0_batch128'
    #bn = 'eggsadam-roi48/eggsadam-roi48_unetv2b_hard-neg-freq10_l1smooth_20190605_165046_adam_lr0.000128_wd0.0_batch128'
    
    #bn = 'eggsadamv2/eggsadamv2-roi48_unetv2b-bn-tanh_hard-neg-freq10_l1smooth_20190613_120327_adam_lr0.000128_wd0.0_batch256'
    #bn = 'eggsadam-stacked/eggsadam-stacked-roi48_unetv2b-tanh_hard-neg-freq10_maxlikelihood_20190614_180709_adam_lr3.2e-05_wd0.0_batch32'
    #bn = 'eggsadam-stacked3/eggsadam-stacked3-roi48_unetv2b-tanh_hard-neg-freq10_maxlikelihood_20190614_205856_adam_lr6e-05_wd0.0_batch60'
    
    #bn = 'eggsadam-stacked/eggsadam-stacked-roi48_unetv2b-tanh_hard-neg-freq10_maxlikelihood_20190614_233317_adam_lr3.2e-05_wd0.0_batch32'
    #bn = 'eggsadamI/eggsadamI-roi48_unetv2b-sigmoid_hard-neg-freq10_maxlikelihoodpooled_20190615_075129_adam_lr0.000128_wd0.0_batch256'
    #bn = 'eggsadamI/eggsadamI-roi96_unetv2b_hard-neg-freq10_maxlikelihoodpooled_20190615_235057_adam_lr3.2e-05_wd0.0_batch64'
    #bn = 'eggsadamI/eggsadamI-roi96_unetv2b_hard-neg-freq10_mixturemodelloss_20190616_190002_adam_lr0.00032_wd0.0_batch64'
    #bn = 'eggsadamI/eggsadamI-roi96_unetv2b_hard-neg-freq10_maxlikelihoodpooled_20190616_170529_adam_lr3.2e-05_wd0.0_batch64'
    
    #bn = 'eggsadamI/eggsadamI-roi96_unetv2b_hard-neg-freq10_mixturemodelloss_20190618_231245_adam_lr0.00032_wd0.0_batch64'
    #bn = 'eggsadamI/eggsadamI-roi96_unetv2b_hard-neg-freq10_maxlikelihoodpooled_20190618_231245_adam_lr3.2e-05_wd0.0_batch64'
    
    #bn = 'eggsadamI/eggsadamI-roi96_unetv2b-bn_hard-neg-freq10_mixturemodelloss_20190619_081519_adam_lr0.00032_wd0.0_batch64'
    bn = 'eggsadamI/eggsadamI-roi96_unetv2b-bn_hard-neg-freq10_maxlikelihoodpooled_20190619_081454_adam_lr3.2e-05_wd0.0_batch64'
    
    n_epoch = None
    
    if n_epoch is None:
        #check_name = 'checkpoint.pth.tar'
        check_name = 'model_best.pth.tar'
    else:
        check_name = f'checkpoint-{n_epoch}.pth.tar'
    
    model_path = Path().home() / 'workspace/localization/results/locmax_detection/eggs' / bn / check_name
    print(model_path)
    
    
    
    #%%
    n_ch_in, n_ch_out  = 1, 1
    batchnorm = '-bn' in bn
    tanh_head = '-tanh' in bn
    sigma_out = '-sigmoid' in bn
    
    if 'unetv2b' in bn:
        model_func = UNetv2B
    else:
        model_func = UNet
    
    if 'maxlikelihood' in bn:
        preeval_func = normalize_softmax
    elif 'mixturemodelloss' in bn:
        preeval_func = reshape_norm
        n_ch_out = n_ch_out*5
    else:
        preeval_func = lambda x : x
    
    model = model_func(n_channels = n_ch_in, 
                       n_classes = n_ch_out, 
                       tanh_head = tanh_head,
                       sigma_out = sigma_out,
                       batchnorm=batchnorm)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    print(state['epoch'])
    #%%
    data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs_first/validation'   
    #data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs/train'
    
    data_root_dir = Path(data_root_dir)
    fnames = data_root_dir.rglob('*.hdf5')
    fnames = list(fnames)
    
#    fnames = []
#    for dd in ['validation', 'test']:
#        data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs/' / dd
#        fnames += list(data_root_dir.rglob('*.hdf5'))
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
        
        
        xhat_n = preeval_func(xhat)
        xout = xhat_n[0].detach().numpy()
        
        
        
        #%%
        bot, top = xout.min(), xout.max()
        xout = (xout - bot) / (top - bot)
        
        #%%
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
        
        axs[0].imshow(img, cmap='gray')
        
        bb = xout[0]
#        if 'maxlikelihood' in bn:
#            bb = cv2.blur(bb, (3,3))
        
        
        axs[1].imshow(bb)#, **norm_args)
        
        coords_pred = peak_local_max(bb, min_distance = 2, threshold_abs = 0.05, threshold_rel = 0.05)
        target = np.array((df['cy'], df['cx'])).T
        #%%
        TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(coords_pred, target, max_dist = 5)
        metrics[0] += (TP, FP, FN)
        
        axs[0].plot(df['cx'], df['cy'], 'xr')
        if coords_pred.size > 0:
            axs[0].plot(coords_pred[...,1], coords_pred[...,0], 'g.')
        #%%
        if 'mixturemodelloss' in bn:
            fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
            axs[0][0].imshow(img, cmap='gray')
            axs[0][0].plot(df['cx'], df['cy'], 'xr')
            if coords_pred.size > 0:
                axs[0][0].plot(coords_pred[...,1], coords_pred[...,0], 'g.')
            
            axs[1][0].imshow(xout[0])#, **norm_args)
            mux = torch.clamp(xhat[:, 1], -3, 3)[0].numpy()
            muy = torch.clamp(xhat[:, 2], -3, 3)[0].numpy()
            sx = torch.clamp(xhat[:, 3], 1, 100)[0].numpy()
            sy = torch.clamp(xhat[:, 4], 1, 100)[0].numpy()
            
            axs[0][1].imshow(mux)
            axs[0][2].imshow(muy)
            axs[1][1].imshow(sx)
            axs[1][2].imshow(sy)
        
        
        
        #%%
        #break
        #%%
        
    plt.show()
    
    
    TP, FP, FN = metrics[0]
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    
    
    print(f'P={P}, R={R}, F1={F1}')
    
    #%%
    print(bn)