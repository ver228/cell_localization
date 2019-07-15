#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:50:47 2019

@author: avelinojaver
"""

from pathlib import Path
import tqdm
import cv2
import numpy as np

if __name__ == '__main__':
    roi_size = 256#(128, 128)
    corner_x = 200
    corner_y = 1
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/'
    root_dir = Path(root_dir)
    
    data_dir = root_dir / 'raw'
    assert root_dir.exists()
    
    save_dir = root_dir / 'crops' / f'x{corner_x}-y{corner_y}-roi{roi_size}'
    save_dir_raw = root_dir / 'crops_raw' / f'x{corner_x}-y{corner_y}-roi{roi_size}'
    
    fnames = data_dir.rglob('*.tif')
    fnames = list(fnames)
    
    
    for ii, fname in tqdm.tqdm(enumerate(fnames)):
        img = cv2.imread(str(fname), -1)
        
        xr, xl = corner_x, corner_x + roi_size
        yr, yl = corner_y, corner_y + roi_size
        
        #%%
        roi1 = img[xr:xl, yr:yl]
        roi2 = img[xr:xl, -yl:-yr]
        
        
        bn = fname.stem
        subfolder = fname.parent.name
        for roi, sub_str in [(roi1, 'POS'),(roi2, 'NEG')]:
            roi_l = np.log(roi.astype(np.float32))
            bot, top = np.min(roi_l), np.max(roi_l)
            roi_n = ((roi_l - bot)/(top-bot)*255).astype(np.uint8)
            
            save_name = save_dir / subfolder / f'ROI-LOG-{sub_str}_{bn}.png'
            save_name.parent.mkdir(exist_ok = True, parents = True)
            cv2.imwrite(str(save_name), roi_n)
            
#            save_name = save_dir_raw / subfolder / f'ROI-RAW-{sub_str}_{bn}.tif'
#            save_name.parent.mkdir(exist_ok = True, parents = True)
#            cv2.imwrite(str(save_name), roi)
            
            
        
    