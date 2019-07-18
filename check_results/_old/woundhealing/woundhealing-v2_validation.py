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
import pandas as pd
import tqdm
import matplotlib.pylab as plt

from cell_localization.evaluation.localmaxima import evaluate_coordinates

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
    scale_int = (0, 4095)
    
    set_type = 'mix+nuclei'#'mix'#'nuclei'#
    results_dir = Path.home() / f'workspace/localization/results/locmax_detection/woundhealing-v2/woundhealing-v2-{set_type}'
    validation_dir = Path.home() / f'workspace/localization/data/woundhealing/annotated/v2/{set_type}/validation'
    
    
    validation_data = []
    for fname in tqdm.tqdm(validation_dir.rglob('*.hdf5')):
        with pd.HDFStore(str(fname), 'r') as fid:
            img = fid.get_node('/img')[:]
            if '/coords' in fid:
                coords_df = fid['/coords']
            else:
                continue
        target = coords_df.loc[:, ['cx', 'cy']].values
        
        x = img.astype(np.float32)
        x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
        X = torch.from_numpy(x[None, None])
        
        validation_data.append((X, target))
        
        
        
    
    scale_int = (0, 4095)
    thresh2check = np.arange(0.05, 1, 0.05)
    
    
    models2check = results_dir.rglob('model_best.pth.tar')
    models2check = sorted(list(models2check))
    
    results = []
    metrics = np.zeros((len(models2check), 3, len(thresh2check)))  
    for i_model, model_path in enumerate(tqdm.tqdm(models2check)):
        model_name = model_path.parent.name
        model, epoch = load_model(model_name)
        #%%
        for X, target in validation_data:
            with torch.no_grad():    
                Xhat = model(X)
            
            xhat = Xhat.squeeze().detach().numpy()
            xr = X.squeeze().detach().numpy()
            
            
            coords_pred = peak_local_max(xhat, 
                                         min_distance = 3, 
                                         threshold_abs = 0.4, 
                                         threshold_rel = 0.1, 
                                         exclude_border = False)
            #coords_pred = peak_local_max(xhat, min_distance = 5, threshold_abs = 0.05, threshold_rel = 0.5)
            coords_pred = coords_pred[:, ::-1]
            
            TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(coords_pred, target, max_dist = 10)
            
            if True:
                _plot_predictions(xr, xhat, coords_pred, target, pred_ind, true_ind)
            
            
            for ith, th in enumerate(thresh2check):
                pred = peak_local_max(xhat, min_distance = 3, threshold_abs = th, threshold_rel = 0.0, exclude_border = False)
                pred = pred[:, ::-1] #match target order
                
                
                TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(pred, target, max_dist = 20)
                metrics[i_model, :, ith] += (TP, FP, FN)
    #%%
    
    
    for model_metric, model_path in zip(metrics, models2check):
        model_name = model_path.parent.name
        fig, axs = plt.subplots(1,2, figsize = (15, 5))
        TP, FP, FN = model_metric
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        axs[0].plot(R, P, '.-')
        axs[1].plot(thresh2check, F1,  '.-')
        
        axs[0].set_xlabel('Recall')
        axs[0].set_ylabel('Precision')
        axs[0].set_xlim((0,1.05))
        axs[0].set_ylim((0,1.05))
        
        axs[1].legend()
        axs[1].set_xlabel('Threshold')
        axs[1].set_ylabel('F1 value')
        axs[1].set_ylim((0, 1.05))
    
        plt.suptitle(model_name)
        
        ibest = np.argmax(F1)
        
        Pb, Rb, F1b = P[ibest], R[ibest], F1[ibest]
        
        print(model_name)
        print(f'P {Pb:.3f} | R {Rb:.3f} | F1 {F1b:.3f}' )
        
        #%%
#        _out = [f'{model_name} ; epoch: {epoch}|']
#        _out += ['image | P | R | F1 |', '-'*30]
#        all_metrics = np.zeros(3)
#        for  TP, FP, FN in sorted(metrics):
#            all_metrics += TP, FP, FN
#            P, R, F1 = _calculate_accuracy(TP, FP, FN)
#            _out += [f' {P:.3} | {R:.3} | {F1:.3} |']
#          
#        P, R, F1 = _calculate_accuracy(*all_metrics)
#        _out += [f'all | {P:.3} | {R:.3} | {F1:.3} |']
#        _out +=  ['-'*30]
#        
#        _out = '\n'.join(_out)
#        print(_out)
#        
#        results.append(_out)
        #%%
        