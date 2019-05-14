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
import pandas as pd
import tqdm
import cv2
import matplotlib.pylab as plt

from cell_localization.evaluation.localmaxima import evaluate_coordinates

from skimage.feature import peak_local_max
#%%

if __name__ == '__main__':
    save_dir =  Path('/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/results/')
    
    
    #bn = 'cell-loc_unet_l1smooth_20190425_160121_adam_lr0.001_wd0.0_batch64'
    #bn = 'cell-loc_unet-bn_l1smooth_20190425_160022_adam_lr0.001_wd0.0_batch64'
    #bn = 'cell-loc_unet-bn_l1smooth_20190425_205040_adam_lr0.001_wd0.0_batch64'
    
    #bn = 'heba/heba_unet-bn_l1smooth_20190501_083712_adam_lr0.001_batch64/heba_unet-bn_l1smooth_20190501_083712_adam_lr0.001_batch64'
    #bn = 'heba/heba_unet_l1smooth_20190501_105726_adam_lr0.001_batch64'
    bn = 'heba/heba_unet_maskfocal_20190501_163856_adam_lr0.001_batch64'
    
    #model_path = Path('/Volumes/loco/') / 'workspace/localization/results/cell_detection' / bn / 'checkpoint.pth.tar'
    
    
    #n_epoch = 399 
    #check_name = f'checkpoint-{n_epoch}.pth.tar'
    check_name = 'checkpoint.pth.tar'
    
    model_path = Path.home() / 'workspace/localization/results/locmax_detection/' / bn / check_name
    #model_path = Path.home() / 'workspace/localization/results/cell_detection' / bn / check_name
    #model_path = Path('/Volumes/loco/') / 'workspace/localization/results/cell_detection' / bn / check_name
   
    scale_int = (0, 4095)
    
    n_ch_in, n_ch_out  = 1, 1
    if '-separated' in bn:
        n_ch_out = 3
    
    batchnorm = '-bn' in bn
         
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm = batchnorm)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    #%%
    #input_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/raw/')
    #fnames = input_dir.glob('*.tif')
    
    input_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/data/validation')
    fnames = input_dir.glob('*.hdf5')
    
    metrics = []
    for fname in tqdm.tqdm(fnames):
        img_id = fname.stem        
        with pd.HDFStore(str(fname), 'r') as fid:
            img = fid.get_node('/img')[:]
            if '/coords' in fid:
                coords_df = fid['/coords']
            else:
                coords_df = None
        
        
        x = img[None].astype(np.float32)
        x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
        
        with torch.no_grad():
            X = torch.from_numpy(x[None])
            Xhat = model(X)
        
        xhat = Xhat.squeeze().detach().numpy()
        xr = x.squeeze()
        
        #%%
        #coords_pred = cv2_peak_local_max(xhat, threshold_relative = 0.1, threshold_abs = 0.05)
        coords_pred = peak_local_max(xhat, min_distance = 5, threshold_abs = 0.05, threshold_rel = 0.5)
        coords_pred = coords_pred[:, ::-1]
        
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = (15, 5))
        axs[0].imshow(xr, cmap='gray')#, vmin=0, vmax=1)
        axs[0].set_title('Original')
        
        #axs[1].imshow(xhat, cmap='gray')#, vmin=0.4)
        axs[1].imshow(xr, cmap='gray')
        axs[1].imshow(xhat, cmap='inferno', alpha=0.5)
        axs[1].set_title('Believe Maps')
        
        axs[2].imshow(xr, cmap='gray')
        #axs[2].plot(coords_pred[..., 0], coords_pred[..., 1], '.r')
        
        #if coords_df is not None:
        #    axs[2].plot(coords_df['cx'], coords_df['cy'], 'x', color='y')
        axs[2].set_title('Predicted Coordinates')
        
        plt.suptitle(fname.stem)
        #%%
        #plt.xlim((800, 1000))
        #plt.ylim((0, 200))
        
        for ax in axs:
            ax.axis('off')
        
        plt.savefig(str(save_dir / f'sample_out_{img_id}.pdf'))
        
        #%%
        target = coords_df.loc[:, ['cx', 'cy']].values
        TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(coords_pred, target)
        
        good = np.zeros(coords_pred.shape[0], np.bool)
        good[pred_ind] = True
        pred_bad = coords_pred[~good]
        
        good = np.zeros(target.shape[0], np.bool)
        good[true_ind] = True
        target_bad = target[~good]
        
        axs[2].plot(pred_bad[:, 0], pred_bad[:, 1], 'x', color = 'r')
        axs[2].plot(target_bad[:, 0], target_bad[:, 1], '.', color = 'r')
        axs[2].plot(coords_pred[pred_ind, 0], coords_pred[pred_ind, 1], 'o', color = 'y')

        #%%
        metrics.append((fname.stem, TP, FP, FN))
        
        
        
        #%%
        
        
    
    for bn, TP, FP, FN in metrics:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        output = [f'{bn}:', 
                  f'precision -> {P}',
                  f'recall -> {R}',
                  f'f1 -> {F1}'
              ]
        print(output)
        
        
    ['7-0:', 'precision -> 0.8224431818181818', 'recall -> 0.6334792122538293', 'f1 -> 0.7156983930778738']
    ['16-0:', 'precision -> 0.755700325732899', 'recall -> 0.7682119205298014', 'f1 -> 0.761904761904762']
    ['1-0:', 'precision -> 0.9034090909090909', 'recall -> 0.9578313253012049', 'f1 -> 0.9298245614035089']