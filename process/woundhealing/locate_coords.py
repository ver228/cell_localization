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


from cell_localization.models import UNet, UNetv2
from cell_localization.trainer import get_device

import cv2
import torch
import numpy as np
import pandas as pd
import tables
import tqdm

from skimage.feature import peak_local_max

filters = tables.Filters(complevel=0, 
                          complib='blosc', 
                          shuffle=True, 
                          bitshuffle=True, 
                          fletcher32=True
                          )

if __name__ == '__main__':
    is_plot = False
    cuda_id = 0
    scale_int = (0, 4095)
    img_src_dir = Path.home() / 'workspace/localization/data/woundhealing/raw/'
    save_dir_root = Path.home() / 'workspace/localization/data/woundhealing/location_predictions'
    
    model_loc_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing/'
    bn = 'woundhealing-all-roi48_unetv2-init-normal_l1smooth_20190531_143703_adam_lr0.000128_wd0.0_batch128'
    model_loc_path = model_loc_dir / bn.partition('_')[0] / bn / 'model_best.pth.tar'
    assert model_loc_path.exists()
    
    
    save_dir = save_dir_root / bn 
    save_dir.mkdir(parents = True, exist_ok = True)
    
    device = get_device(cuda_id)
    
    n_ch_in, n_ch_out  = 1, 1
    model_loc = UNetv2(n_channels = n_ch_in, 
                 n_classes = n_ch_out, 
                 batchnorm = False,
                 init_type = None)
    
    
    state = torch.load(model_loc_path, map_location = 'cpu')
    model_loc.load_state_dict(state['state_dict'])
    model_loc.eval()
    model_loc = model_loc.to(device)
    
    #%%
    img_paths = img_src_dir.rglob('*.tif')
    img_paths = list(img_paths)
    for img_path in tqdm.tqdm(img_paths):
        img = cv2.imread(str(img_path), -1)
        
        x = img.astype(np.float32)
        x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
        
        with torch.no_grad():
            X = torch.from_numpy(x[None, None])
            X = X.to(device)
            
            Xhat = model_loc(X)
            
            pred_loc_map = Xhat.squeeze().cpu().detach().numpy()
            
        
        coords_pred = peak_local_max(pred_loc_map, min_distance = 3, threshold_abs = 0.1, threshold_rel = 0.1)
        coords_pred = coords_pred[:, ::-1]
        df = pd.DataFrame({'cx':coords_pred[:, 0], 
                            'cy':coords_pred[:, 1]})
        
    
        
        base_dir = str(img_path.parent).replace(str(img_src_dir), str(save_dir))
        save_name = Path(base_dir) / f'{img_path.name}_preds.csv'
        save_name.parent.mkdir(parents = True, exist_ok = True)
        
        
        df.to_csv(save_name, index=False)
        