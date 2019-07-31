#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:03:09 2019

@author: avelinojaver
"""

import cv2
import pandas as pd
from pathlib import Path
import tables
import tqdm

#%%


filters = tables.Filters(complevel=0, 
                          complib='blosc', 
                          shuffle=True, 
                          bitshuffle=True, 
                          fletcher32=True
                          )

if __name__ == '__main__':
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/raw_separate_channels/'
    root_dir = Path(root_dir)
    
    save_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/splitted/'
    
    
    
    
    GT_noisy = pd.read_excel(root_dir / 'groundTruthForAvelino_30.xlsx')
    img_ids = pd.read_excel(root_dir / 'imageIDsForAvelino_30.xlsx')
    
    resize_factor = 0.5
    save_dir = save_dir / f'F{resize_factor}x'
    save_dir.mkdir(parents = True, exist_ok = True)
    
    missing_paths = []
    for _, row in tqdm.tqdm(img_ids.iterrows()):
        img_id = row['ImageNumber']
        coords = GT_noisy[GT_noisy['ImageNumber'] == img_id]
        
        img_nuclei_path = root_dir / 'nuclei' / row['Nuclear']
        img_actin_path = root_dir / 'membrane' / row['Actin']
        
        is_missing = False
        if not img_nuclei_path.exists():
            missing_paths.append(img_nuclei_path)
            is_missing = True
        if not img_actin_path.exists():
            missing_paths.append(img_actin_path)
            is_missing = True
        
        if is_missing:
            continue
        
        coords2save = pd.DataFrame({'type_id':1, 
                                'cx': coords['Location_Center_X']*resize_factor, 
                                'cy': coords['Location_Center_Y']*resize_factor
                                 }
                                )
        coords2save = coords2save.to_records(index=False)    
        
        img1 = cv2.imread(str(img_nuclei_path), -1)
        img2 = cv2.imread(str(img_actin_path), -1)
        
        img1 = cv2.resize(img1, (0,0), fx = resize_factor, fy = resize_factor)
        img2 = cv2.resize(img2, (0,0), fx = resize_factor, fy = resize_factor)
        
        #print(img1.max(), img2.max())
        #continue
        bn = img_nuclei_path.stem
        save_name = save_dir / (bn + '.hdf5')
        with tables.File(str(save_name), 'w') as fid:
            fid.create_carray('/', 'img1', obj = img1, filters  = filters)
            fid.create_carray('/', 'img2', obj = img2, filters  = filters)
            fid.create_table('/', 'coords', obj = coords2save)
                        
                      