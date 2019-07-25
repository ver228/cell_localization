#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:55:24 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'scripts') )

import numpy as np
from cell_localization.trainer import get_device
from cell_localization.evaluation.localmaxima import score_coordinates


from load_model import load_model


import tqdm
import matplotlib.pylab as plt

if __name__ == '__main__':
   cuda_id = 0
   device = get_device(cuda_id)
    
   #bn = 'worm-eggs-adam+Feggs+roi96_unet-simple_l2-G1.5_20190717_162118_adam_lr0.000128_wd0.0_batch128'
   #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-1_unet-simple_l2-G1.5_20190717_162710_adam_lr0.000128_wd0.0_batch128'
   
   #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/eggs'
   
   #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/eggs'
   
   #bn = 'woundhealing-v2-nuclei+Fwoundhealing+roi96_unet-simple_maxlikelihood_20190718_221552_adam_lr0.000128_wd0.0_batch128'
   #bn = 'woundhealing-v2-mix+Fwoundhealing+roi96_unet-simple_maxlikelihood_20190718_222136_adam_lr0.000128_wd0.0_batch128'
   #bn = 'woundhealing-v2-mix+Fwoundhealing+roi96_unet-simple_l2-reg-G2.5_20190719_145851_adam_lr0.000128_wd0.0_batch128'
   #bn = 'woundhealing-v2-nuclei+Fwoundhealing+roi48_unet-simple_l1-reg-G1.5_20190719_194158_adam_lr0.000256_wd0.0_batch256'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/'
   
   #bn = 'eosinophils-20x+Feosinophils+roi64_unet-simple_maxlikelihood_20190723_233841_adam_lr0.000256_wd0.0_batch256'
   #bn = 'eosinophils-20x+Feosinophils+roi48_unet-simple_maxlikelihood_20190724_080105_adam_lr0.000256_wd0.0_batch256'
   bn = 'eosinophils-20x+Feosinophilsonly+roi48+hard-neg-1_unet-simple_maxlikelihood_20190724_080046_adam_lr0.000256_wd0.0_batch256'
   model_dir = Path.home() / 'workspace/localization/results/locmax_detection/eosinophils/20x/'
   
   #bn = 'limphocytes-20x+Flimphocytes+roi64_unet-simple_maxlikelihood_20190723_233842_adam_lr0.000256_wd0.0_batch256'
   #bn = 'limphocytes-20x+Flimphocytesonly+roi48+hard-neg-1_unet-simple_maxlikelihood_20190724_080048_adam_lr0.000256_wd0.0_batch256'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/limphocytes/20x/'
   
   #data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs_first' 
   data_root_dir = None
   
   model, data_flow, epoch = load_model(model_dir, bn, data_root_dir = data_root_dir, checkpoint_name = 'model_best.pth.tar')
   model = model.to(device)
    
   #%%
   N = len(data_flow.data_indexes)
   for ind in tqdm.trange(N):
        
        image, target = data_flow.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = image.to(device)
   
        predictions, belive_maps = model(image[None])
        xr = image.squeeze().detach().cpu().numpy()
        pred_coords = predictions[0]['coordinates'].detach().cpu().numpy()
        true_coords = target['coordinates'].detach().cpu().numpy()
         
        TP, FP, FN, pred_ind, true_ind = score_coordinates(pred_coords, true_coords, max_dist = 5)
        
        img = image.detach().cpu().numpy()
        if image.shape[0] == 3:
            img = np.rollaxis(img, 0, 3)
            img = img[..., ::-1]
        else:
            img = img[0]
        
        mm = belive_maps[0, 0].detach().cpu().numpy()
        
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize = (15, 5))
        
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Raw Image')
        axs[1].imshow(mm)
        if pred_ind is None:
            axs[1].plot(pred_coords[:, 0], predictions[:, 1], 'x', color = 'r')
            axs[1].plot(target[:, 0], target[:, 1], '.', color = 'r')
            
        else:
            good = np.zeros(pred_coords.shape[0], np.bool)
            good[pred_ind] = True
            pred_bad = pred_coords[~good]
            
            good = np.zeros(true_coords.shape[0], np.bool)
            good[true_ind] = True
            target_bad = true_coords[~good]
            
            axs[0].plot(pred_bad[:, 0], pred_bad[:, 1], 'x', color = 'r')
            axs[0].plot(target_bad[:, 0], target_bad[:, 1], '.', color = 'r')
            axs[0].plot(pred_coords[pred_ind, 0], pred_coords[pred_ind, 1], 'o', color='g')
            
       