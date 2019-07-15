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


from cell_localization.models import UNet, UNetFlatter, UNetv2, UNetscSE, UNetv2B, UNetAttention
import torch
import numpy as np
import cv2
import tqdm
import matplotlib.pylab as plt
import torch.nn.functional as F

from skimage.feature import peak_local_max
#%%
_is_plot = True

def _calculate_accuracy(TP, FP, FN):
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    return P, R, F1


def load_model(bn):
    model_name = bn.split('_')[1]
    
    n_ch_in, n_ch_out  = 1, 1
    batchnorm = '-bn' in model_name
    tanh_head = '-tanh' in model_name
    sigma_out = '-sigmoid' in model_name
    try:
        ll = model_name.split('-')
        ii = ll.index('init')
        init_type = ll[ii+1]
    except ValueError:
        init_type = 'xavier'
    
    
    ss = model_name.partition('-')[0]
    model_args = dict(
            n_channels = n_ch_in, 
             n_classes = n_ch_out, 
             batchnorm = batchnorm,
             tanh_head = tanh_head,
             sigma_out = sigma_out,
             init_type = init_type
            )
    
    if ss == 'unetv2b':
        model = UNetv2B(**model_args)
    elif ss == 'unetv2':
        model = UNetv2(**model_args)
    elif ss == 'unetscSE':
        model = UNetscSE(**model_args)
    elif ss == 'unetattn':
        model = UNetAttention(**model_args)
    
    elif ss == 'unetflatter':
        model = UNetFlatter(**model_args)
    elif ss == 'unet':
        model = UNet(**model_args)
    else:
        raise ValueError('Not implemented `{}`'.format(ss))
    
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    return model, state['epoch']

def _plot_predictions(img, belive_maps, coords_pred, target, pred_ind, true_ind):
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (15, 5))
    axs[0].imshow(img, cmap='gray')#, vmin=0, vmax=1)
    axs[0].set_title('Original')
    
    axs[1].imshow(img, cmap='gray')
    axs[1].imshow(belive_maps, cmap='inferno', alpha=0.5)
    axs[1].set_title('Believe Maps')
    
    axs[2].imshow(img, cmap='gray')
    axs[2].set_title('Predicted Coordinates')
    
    plt.suptitle(fname.stem)
    
    for ax in axs:
        ax.axis('off')
    

    good = np.zeros(coords_pred.shape[0], np.bool)
    good[pred_ind] = True
    pred_bad = coords_pred[~good]
    
    good = np.zeros(target.shape[0], np.bool)
    good[true_ind] = True
    target_bad = target[~good]
    
    axs[2].plot(pred_bad[:, 0], pred_bad[:, 1], 'x', color = 'r')
    axs[2].plot(target_bad[:, 0], target_bad[:, 1], '.', color = 'r')
    axs[2].plot(coords_pred[pred_ind, 0], coords_pred[pred_ind, 1], 'o', color = 'y')
    
    
    plt.suptitle(model_name)


if __name__ == '__main__':
    root_dir = Path.home() / f'OneDrive - Nexus365/heba'
    model_path = root_dir / 'cell_localization_simple/models/mix/woundhealing-v2-mix-roi96_unetv2b-init-normal_l1smooth_20190703_000218_adam_lr0.000128_wd0.0_batch128/model_best.pth.tar'
    src_dir =  root_dir / 'WoundHealing/raw/mix/'
    
    scale_int = (0, 4095)
    model_name = model_path.parent.name
    model, epoch = load_model(model_name)
   
    validation_data = []
    for fname in tqdm.tqdm(src_dir.rglob('*.tif')):
        img = cv2.imread(str(fname), -1)
        x = img.astype(np.float32)
        x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
        X = torch.from_numpy(x[None, None])
        
        with torch.no_grad():    
            Xhat = model(X)
        
        xhat = Xhat.squeeze().detach().numpy()
        xr = X.squeeze().detach().numpy()
        
        #%%
        coords_pred = peak_local_max(xhat, 
                                     min_distance = 3, 
                                     threshold_abs = 0.4, 
                                     threshold_rel = 0.0, 
                                     exclude_border = False)
        
        
        plt.figure()
        plt.imshow(img, cmap = 'gray')
        plt.plot(coords_pred[:, 1], coords_pred[:, 0], '.r')
        #%%
        break
        #%%
        threshold_abs = 0.4
        min_distance = 3
        exclude_border = True
        
        kernel_size = 2 * min_distance + 1
        x_in = Xhat
        
        
        x_max = F.max_pool2d(x_in, kernel_size, stride = 1, padding = kernel_size//2)
        x_mask = (x_max == x_in)
        if exclude_border:
            exclude = 2 * min_distance
            x_mask[..., :exclude] = x_mask[..., -exclude:] = 0
            x_mask[..., :exclude, :] = x_mask[..., -exclude:, :] = 0
        x_mask &= x_in > threshold_abs
        
        
        results = []
        for xi, xm in zip(x_in, x_mask):
            ind = xm.nonzero()
            res = {}
            res['scores'] = xi[ind[:, 0], ind[:, 1], ind[:, 2]]
            res['labels'] = ind[:, 0]
            res['coodinates'] = ind[:, [2, 1]]
            results.append(res)
            
        
        for xi, res in zip(X, results):
            coords = res['coodinates'].detach().cpu().numpy()
            
            plt.imshow(xi[0])
            plt.plot(coords[:, 0], coords[:, 1], '.')
            
            
        
     
        #%%
            
        
        
        #x_nms = x_max*x_nms.float()
        
        
        
        #plt.imshow(x_nms[0,0])
        
        
        
        
    #%%
#    size = 2 * min_distance + 1
#    remove = (footprint.shape[i] if footprint is not None
#                      else 2 * exclude_border)
#            mask[:remove // 2] = mask[-remove // 2:] = False