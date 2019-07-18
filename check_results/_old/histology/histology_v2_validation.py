#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:55:24 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))

import torch
import pandas as pd

import numpy as np
import matplotlib.pylab as plt
import tqdm

from skimage.feature import peak_local_max
from cell_localization.models import UNet, UNetv2, UNetv2B
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
    
    #bn = 'bladder-tiles-40x_unet_l1smooth_20190523_221910_adam_lr6.4e-05_batch64'
    #bn = 'bladder-tiles-40x_unet_hard-neg-freq1_l1smooth_20190524_174327_adam_lr6.4e-05_batch64'
    
    #bn = 'bladder-tiles-20x_unet_l1smooth_20190523_221913_adam_lr6.4e-05_batch64'
    #bn = 'bladder-tiles-20x_unet_hard-neg-freq1_l1smooth_20190524_172750_adam_lr6.4e-05_batch64'
    #bn = 'bladder-tiles-40x_unetv2-norm_l1smooth_20190529_120040_adam_lr6.4e-05_wd0.0_batch64'
    
    
    bn = 'bladder-tiles-roi64-20x/bladder-tiles-roi64-20x_unetv2_l1smooth_20190529_193223_adam_lr6.4e-05_wd0.0_batch64'
    
    #bn = 'bladder-tiles-20x/bladder-tiles-20x-roi64_unetv2b_maxlikelihoodpooled_20190621_152345_adam_lr0.000256_wd0.0_batch256'
    #bn = 'bladder-tiles-20x/bladder-tiles-20x-roi64_unetv2b_hard-neg-freq5_maxlikelihoodpooled_20190621_134250_adam_lr0.000256_wd0.0_batch256'
    
    is_switch_channels = True
    
    set_type = bn.partition('_')[0]
    
    #set_type = '-'.join([x for x in set_type.split('-') if not x.startswith('roi')])
    magnification = '20x' if '20x' in bn else '40x'
    
    n_epoch = None
    #n_epoch = 99
    
    if n_epoch is None:
        #check_name = 'checkpoint.pth.tar'
        check_name = 'model_best.pth.tar'
    else:
        check_name = f'checkpoint-{n_epoch}.pth.tar'
    
    #model_path = Path().home() / 'workspace/localization/results/histology_detection' / bn / check_name
    #model_path = Path.home() / 'workspace/localization/results/locmax_detection/bladder' / magnification / set_type / bn / check_name
    model_path = Path.home() / 'workspace/localization/results/locmax_detection/bladder' / magnification / bn / check_name
    
    n_ch_in, n_ch_out  = 3, 2
    batchnorm = 'unet-bn' in bn
    
    if 'unetv2b' in bn:
        model_func = UNetv2B
    elif 'unetv2' in bn:
        model_func = UNetv2
    else:
        model_func = UNet
    model = model_func(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm=batchnorm)
    
    if 'maxlikelihood' in bn:
        preeval_func = normalize_softmax
    elif 'mixturemodelloss' in bn:
        preeval_func = reshape_norm
        n_ch_out = n_ch_out*5
    else:
        preeval_func = lambda x : x
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    #%%
    val_type = 'validation'
    #val_type = 'test'
    #val_type = 'train'#'test'#
    
    data_root_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/' / magnification / val_type
    #data_root_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/' / margnification / val_type
    
    
    data_root_dir = Path(data_root_dir)
    assert data_root_dir.exists()
    
    save_dir = f'/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/results/{bn}/'
    
    data_root_dir = Path(data_root_dir)
    fnames = data_root_dir.rglob('*.hdf5')
    
    
    fnames = list(fnames)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok = True)
    
    norm_args = dict(vmin = 0, vmax=1)
    
    data_info = {'Lymphocytes' : (0, 'y'), 'Eosinophils': (1, 'c')}
    
    
    thresh2check = np.arange(0.05, 1, 0.05)
    th_ind2eval = 1
    
    metrics = np.zeros((len(data_info), 3, len(thresh2check)))  
      
    for iname, fname in enumerate(tqdm.tqdm(fnames)):
        with pd.HDFStore(str(fname), 'r') as fid:
            img = fid.get_node('/img')[:]
            if not '/coords' in fid:
                continue
            
            df = fid['/coords']
        
        xin = np.rollaxis(img, 2, 0)
        if not is_switch_channels:
            xin = xin[::-1]
        
        xin = xin.astype(np.float32)/255
        
        with torch.no_grad():
            xin = torch.from_numpy(xin[None])
            xhat = model(xin)
        
        
        xhat_n = preeval_func(xhat)
        xout = xhat_n[0].detach().numpy()
        
        for t_lab, (ind, color) in data_info.items():
            dat = df[df['type'] == t_lab]
            mm = xout[ind]
            target = dat.loc[:, ['cx', 'cy']].values
            
            for ith, th in enumerate(thresh2check):
                pred = peak_local_max(mm, min_distance = 3, threshold_abs = th, threshold_rel = 0.0)
                pred = pred[:, ::-1] #match target order
                
                TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(pred, target, max_dist = 20)
                metrics[ind, :, ith] += (TP, FP, FN)
        
        #%%
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (15, 5))
        fig.suptitle(fname.name)

        if is_switch_channels:
            img = img[..., ::-1]
        
        axs[0].imshow(img)
        axs[0].set_title('Raw Image')
        axs[1].imshow(xout[0])
        axs[2].imshow(xout[1])
        
        for t_lab, (ind, color) in data_info.items():
            dat = df[df['type'] == t_lab]
            mm = xout[ind]
            target = dat.loc[:, ['cx', 'cy']].values
            
            th = thresh2check[th_ind2eval]
            pred = peak_local_max(mm, min_distance = 3, threshold_abs = th, threshold_rel = 0.0)
            pred = pred[:, ::-1]
            TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(pred, target, max_dist = 20)
            
            if pred_ind is None:
                axs[ind + 1].plot(pred[:, 0], pred[:, 1], 'x', color = 'r')
                axs[ind + 1].plot(target[:, 0], target[:, 1], '.', color = 'r')
                
            else:
                good = np.zeros(pred.shape[0], np.bool)
                good[pred_ind] = True
                pred_bad = pred[~good]
                
                good = np.zeros(target.shape[0], np.bool)
                good[true_ind] = True
                target_bad = target[~good]
                
                axs[ind + 1].plot(pred_bad[:, 0], pred_bad[:, 1], 'x', color = 'r')
                axs[ind + 1].plot(target_bad[:, 0], target_bad[:, 1], '.', color = 'r')
                axs[ind + 1].plot(pred[pred_ind, 0], pred[pred_ind, 1], 'o', color=color)
                axs[ind + 1].set_title(t_lab)
                
    #%%
    fig, axs = plt.subplots(1,2, figsize = (15, 5))
    for metric, t_lab in zip(metrics, data_info.keys()):
        TP, FP, FN = metric
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        axs[0].plot(R, P, '.-', label = t_lab)
        axs[1].plot(thresh2check, F1,  '.-', label = t_lab)
        
    axs[0].set_xlabel('Recall')
    axs[0].set_ylabel('Precision')
    axs[0].set_xlim((0,1.05))
    axs[0].set_ylim((0,1.05))
    
    axs[1].legend()
    axs[1].set_xlabel('Threshold')
    axs[1].set_ylabel('F1 value')
    axs[1].set_ylim((0, 1.05))
    
    plt.suptitle(bn)
            #%%
    results = []
    
    for lab_t, (ind, _) in data_info.items():
        TP, FP, FN = metrics[ind, :, th_ind2eval]
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        
        output = [f'{lab_t}:', 
                  f'precision -> {P}',
                  f'recall -> {R}',
                  f'f1 -> {F1}'
              ]
        results.append('\n'.join(output))
    str2save = '\n----\n'.join(results)
    
    th = thresh2check[th_ind2eval]
    print(f'Evaluated as >{th}')
    print(str2save)

    