#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:15:38 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))

from cell_localization.models import UNet
from cell_localization.trainer import get_device

from skimage.feature import peak_local_max
import tqdm
import tables
import torch
import numpy as np
import pandas as pd

if __name__ == '__main__':
    
    #where the masked files are located
    root_dir = Path.home() / 'workspace/WormData/screenings/Drug_Screening/MaskedVideos/'
    
    #where the pre-trained model is located
    model_path = Path().home() / 'workspace/localization/results/locmax_detection/eggs-int/eggs-int_unet_hard-neg-freq1_l1smooth_20190513_081902_adam_lr0.000128_batch128/model_best.pth.tar'
    
    #where the predictions are going to be stored
    save_dir = Path.home() / 'workspace/localization/predictions/worm_eggs/' / model_path.parent.name
    
    #%%
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)
    
    cuda_id = 0
    device = get_device(cuda_id)
    
    
    n_ch_in, n_ch_out  = 1, 1
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model = model.to(device)
    model.eval()
    
    mask_files = root_dir.rglob('*.hdf5')
    mask_files = [x for x in mask_files if not x.name.startswith('.')]
    for mask_file in tqdm.tqdm(mask_files):
        with tables.File(mask_file, 'r') as fid:
            imgs = fid.get_node('/full_data')[:]
        
        
        preds_l = []
        for frame_number, img in enumerate(imgs):
            xin = img[None]
            xin = xin.astype(np.float32)/255
            
            with torch.no_grad():
                xin = torch.from_numpy(xin[None])
                xin = xin.to(device)
                xhat = model(xin)
            
            xout = xhat[0].detach().cpu().numpy()
            
            coords_pred = peak_local_max(xout[0], min_distance = 2, threshold_abs = 0.05, threshold_rel = 0.1)
            
            preds_l += [('/full_data', frame_number, *cc) for cc in coords_pred]
        
        preds_df = pd.DataFrame(preds_l, columns = ['group_name', 'frame_number', 'x', 'y'])
        
        save_name = Path(str(mask_file).replace(str(root_dir), str(save_dir)))
        save_name = save_name.parent / (save_name.stem + '_eggs-preds.csv')
        save_name.parent.mkdir(exist_ok=True, parents=True)
        
        preds_df.to_csv(save_name, index = False)
    