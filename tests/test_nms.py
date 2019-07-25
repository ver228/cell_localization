#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 11:04:58 2019

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'scripts') )


from cell_localization.trainer import get_device
from cell_localization.evaluation.localmaxima import score_coordinates


from load_model import load_model


import tqdm
import matplotlib.pylab as plt

if __name__ == '__main__':
   cuda_id = 0
   device = get_device(cuda_id)
    
   #bn = 'woundhealing-v2-mix+Fwoundhealing+roi96_unet-simple_l1smooth-G1.5_20190717_115942_adam_lr6.4e-05_wd0.0_batch64'
   #bn = 'woundhealing-v2-mix+Fwoundhealing+roi48_unet-simple_l2-G1.5_20190717_152330_sgd+stepLR-4-0.1_lr0.000256_wd0.0_batch256'
   #bn = 'woundhealing-v2-mix+Fwoundhealing+roi48_unet-simple_l2-G1.5_20190717_161240_adam_lr0.000128_wd0.0_batch128'
   
   
   model_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2'
   
   model, data_flow = load_model(model_dir, bn)
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
       #%%
       fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
       axs[0].imshow(xr)
       axs[0].plot(pred_coords[:, 0], pred_coords[:, 1], 'r.')
       #plt.plot(true_coords[:, 0], true_coords[:, 1], 'rx')
       
       m = belive_maps[0,0].detach().cpu().numpy()
       axs[1].imshow(m)