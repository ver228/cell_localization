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


from cell_localization.models import UNet, UNetFlatter, UNetv2, UNetscSE
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

if __name__ == '__main__':
    
    root_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing/'
    #validation_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/data/validation')
    validation_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/all/validation/'
    
    validation_names_d = {
                        'nuclei+membrane':'8-0.hdf5',
                        'nuclei':'1-0.hdf5',
                        'membrane':'16-0.hdf5'
                        }
    
    scale_int = (0, 4095)
    models2check = root_dir.rglob('model_best.pth.tar')
    models2check = list(models2check)
    
    results = []
    for model_path in tqdm.tqdm(models2check):
        model_name = model_path.parent.name
        
        
        n_ch_in, n_ch_out  = 1, 1
        if '-separated' in model_name:
            n_ch_out = 3
        
        batchnorm = '-bn' in model_name
        patchnorm = '-patchnorm' in model_name
        
        if 'unetflatterv2' in model_name:
            from functools import partial
            model_func = partial(UNetFlatter, interp_mode = 'bilinear')
        elif 'unetflatter' in model_name:
            model_func = UNetFlatter
        elif 'unetv2' in model_name:
            model_func = UNetv2
        elif 'unetscSE' in model_name:
            model_func = UNetscSE
        else:
            model_func = UNet
        
        
        model = model_func(n_channels = n_ch_in, 
                     n_classes = n_ch_out, 
                     batchnorm = batchnorm,
                     init_type = None)
        
        
        state = torch.load(model_path, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
        model.eval()
        
        
        
        metrics = []
        for img_type, img_name in validation_names_d.items():
            fname = validation_dir / img_name
            with pd.HDFStore(str(fname), 'r') as fid:
                img = fid.get_node('/img')[:]
                if '/coords' in fid:
                    coords_df = fid['/coords']
                else:
                    coords_df = None
            
            
            x = img[None].astype(np.float32)
            
            if patchnorm:
                x -= x.mean()
                x -= x.std()
            else:
                x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
            
            with torch.no_grad():
                X = torch.from_numpy(x[None])
                Xhat = model(X)
            
            xhat = Xhat.squeeze().detach().numpy()
            xr = x.squeeze()
            
            
            coords_pred = peak_local_max(xhat, min_distance = 3, threshold_abs = 0.1, threshold_rel = 0.1)
            #coords_pred = peak_local_max(xhat, min_distance = 5, threshold_abs = 0.05, threshold_rel = 0.5)
            coords_pred = coords_pred[:, ::-1]
            
            target = coords_df.loc[:, ['cx', 'cy']].values
            TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(coords_pred, target, max_dist = 10)
            metrics.append((img_type, TP, FP, FN))
            
            if _is_plot:
                #%%
                fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (15, 5))
                axs[0].imshow(xr, cmap='gray')#, vmin=0, vmax=1)
                axs[0].set_title('Original')
                
                axs[1].imshow(xr, cmap='gray')
                axs[1].imshow(xhat, cmap='inferno', alpha=0.5)
                axs[1].set_title('Believe Maps')
                
                axs[2].imshow(xr, cmap='gray')
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
        #%%
        _out = [f'{model_name} ; epoch: {state["epoch"]}|']
        _out += ['image | P | R | F1 |', '-'*30]
        all_metrics = np.zeros(3)
        for metric_name, TP, FP, FN in sorted(metrics):
            all_metrics += TP, FP, FN
            P, R, F1 = _calculate_accuracy(TP, FP, FN)
            _out += [f'{metric_name} | {P:.3} | {R:.3} | {F1:.3} |']
          
        P, R, F1 = _calculate_accuracy(*all_metrics)
        _out += [f'all | {P:.3} | {R:.3} | {F1:.3} |']
        _out +=  ['-'*30]
        
        _out = '\n'.join(_out)
        print(_out)
        
        results.append(_out)
        #%%
        