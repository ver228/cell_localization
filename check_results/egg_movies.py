#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 22:04:53 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'scripts') )
#%%
from cell_localization.trainer import get_device


from load_model import load_model

import cv2
import tqdm
import matplotlib.pylab as plt
import tables
import torch
import numpy as np
#%%
if __name__ == '__main__':
   cuda_id = 0
   device = get_device(cuda_id)
    
   #bn = 'worm-eggs-adam+Feggs+roi96_unet-simple_l2-G1.5_20190717_162118_adam_lr0.000128_wd0.0_batch128'
   #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-1_unet-simple_l2-G1.5_20190717_162710_adam_lr0.000128_wd0.0_batch128'
   
   #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/eggs'
   
   bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
   model_dir = Path.home() / 'workspace/localization/results/locmax_detection/eggs'
   
   #data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs_first' 
   data_root_dir = None
   
   model, data_flow = load_model(model_dir, bn, data_root_dir = data_root_dir, checkpoint_name = 'model_best.pth.tar')
   model = model.to(device)
   #%%
   #video_file = Path('/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/tierpsy_test_data/single_worm/MaskedVideos/N2 on food L_2010_02_26__08_44_59___7___1.hdf5')
   
   video_file = Path('/Users/avelinojaver/OneDrive - Imperial College London/tierpsy_examples/tierpsy_test_data/single_worm/MaskedVideos/N2 on food R_2011_09_13__11_59___3___3.hdf5')
   with tables.File(video_file, 'r') as fid:
       masks = fid.get_node('/mask')
       microns_per_pixel = masks._v_attrs['microns_per_pixel']
       
       img_ori = masks[14515]
       q = np.percentile(img_ori[img_ori>0], 99)
       img_ori[img_ori==0] = q
       
       hh = microns_per_pixel/10
       img = cv2.resize(img_ori, (0,0), fx = hh, fy = hh)
       
   
   x = img.astype(np.float32)/255
   x = torch.from_numpy(x[None, None])
   
   
   predictions, belive_maps = model(x)
   
   mm = belive_maps[0,0].detach().cpu().numpy()
   preds = predictions[0]
   coords = preds['coordinates'].detach().cpu().numpy()
   scores = preds['scores'].detach().cpu().numpy()
   
   fig, axs = plt.subplots(1, 2, figsize = (15, 10), sharex = True, sharey = True)
   axs[0].imshow(img, cmap='gray')
   axs[0].plot(coords[:, 0], coords[:, 1], '.r')
   axs[1].imshow(mm, cmap='gray')
       
       
       
       