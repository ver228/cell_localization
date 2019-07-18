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

from skimage.feature import peak_local_max
from cell_localization.models import UNet
from cell_localization.evaluation.localmaxima import evaluate_coordinates

if __name__ == '__main__':
    #bn = 'bladder-cancer-tils_unet_l1smooth_20190405_113107_adam_lr0.00064_wd0.0_batch64'
    #bn = 'bladder-cancer-tils_unet_l1smooth_20190406_000552_adam_lr0.00064_wd0.0_batch64'
    
    #bn = 'bladder-cancer-tils-40x_unet_l1smooth_20190416_211621_adam_lr0.00064_wd0.0_batch64'
    #bn = 'bladder-cancer-tils-40x_unet-bn_adjl1smooth_20190426_201005_adam_lr0.0064_wd0.0_batch64'

    bn = 'bladder-40x/bladder-40x_unet_l1smooth_20190501_105758_adam_lr0.00064_batch64'    
    #bn = 'bladder-40x/bladder-40x_unet_l1smooth_20190508_152832_adam_lr0.000128_batch128'
    
    n_epoch = None
    #n_epoch = 99
    
    if n_epoch is None:
        #check_name = 'checkpoint.pth.tar'
        check_name = 'model_best.pth.tar'
    else:
        check_name = f'checkpoint-{n_epoch}.pth.tar'
    
    #model_path = Path().home() / 'workspace/localization/results/histology_detection' / bn / check_name
    model_path = Path.home() / 'workspace/localization/results/locmax_detection/' / bn / check_name
    
    
    n_ch_in, n_ch_out  = 3, 2
    batchnorm = 'unet-bn' in bn
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm=batchnorm)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    #%%
    #data_root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/rois/train/'
    #data_root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/rois/test/'
    #data_root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/rois/40x/test/'
    
    data_root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/rois/40x/validation/'
    #data_root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/rois/20x/validation/'
    
    #data_root_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/40x/validation/'
    
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
    
    metrics = np.zeros((len(data_info), 3))
    
    for iname, fname in enumerate(tqdm.tqdm(fnames)):
        #if iname > 10:
        #    break
        
        with pd.HDFStore(str(fname), 'r') as fid:
            img = fid.get_node('/img')[:]
            df = fid['/coords']
        
        xin = np.rollaxis(img, 2, 0)
        xin = xin.astype(np.float32)/255
        
        with torch.no_grad():
            xin = torch.from_numpy(xin[None])
            xhat = model(xin)
        
        
        xout = xhat[0].detach().numpy()
        #%%
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (15, 5))
        
        axs[0].imshow(img)
        axs[1].imshow(xout[0])
        axs[2].imshow(xout[1])
        for t_lab, (ind, color) in data_info.items():
            
            dat = df[df['type'] == t_lab]
            mm = xout[ind]
            
            target = dat.loc[:, ['cx', 'cy']].values
            
            #pred = cv2_peak_local_max(mm, threshold_abs = 0.05, threshold_relative = 0.1)
            pred = peak_local_max(mm, min_distance = 5, threshold_abs = 0.05, threshold_rel = 0.1)
            pred = pred[:, ::-1]
            
            if pred.size > 0:
                ii = pred.shape[0]//2
                
                x, y = pred[ii]
                s = 100
                
                x = max(x, s)
                y = max(y, s)
                
                x = min(x, img.shape[0] - s)
                y = min(y, img.shape[1] - s)
                
                
                lx = (x - s, x + s)
                ly = (y - s, y + s)
                
                TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(pred, target, max_dist = 20)
                
                metrics[ind, :] += (TP, FP, FN)
                
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
            
        axs[0].set_title('Raw Image')
        
        #%%
#        #%%
#        for ax in axs:
#            ax.set_xlim(lx)
#            ax.set_ylim(ly)
#            ax.axis('off')
#        fig.savefig(save_dir / f'{fname.stem}.pdf')
        
    plt.show()
    #%%
    results = []
    for lab_t, (ind, _) in data_info.items():
        TP, FP, FN = metrics[ind]
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
    print(str2save)