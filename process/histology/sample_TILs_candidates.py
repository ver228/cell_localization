#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:34:04 2019

@author: avelinojaver
"""
from pathlib import Path 

import random
import pandas as pd
import numpy as np
import tqdm
import cv2

from openslide import OpenSlide

if __name__ == '__main__':
    slides_dir = Path().home() / 'projects/bladder_cancer_tils/raw/'
    
    roi_size_ori = 512
    top_n = 100
    top_k = 1
    only_limphocytes = False
    ignore_high_density = True
    
    
    min_counts = {'L':20, 'E':5}
    
#    roi_size_ori = 4096
#    top_n = 8
#    top_k = 3
#    only_limphocytes = True
#    ignore_high_density = False
    
    bn = 'bladder-tiles-roi64-20x_unetv2_l1smooth_20190529_193223_adam_lr6.4e-05_wd0.0_batch64'
    coords_dir = Path.home() / 'workspace/localization/predictions/histology_detection' / bn
    coords_files = coords_dir.rglob('*.csv')
    coords_files = list(coords_files)
    
    slides_fnames = [x for x in slides_dir.rglob('*.svs') if not x.name.startswith('.')]
    slides_fnames += [x for x in slides_dir.rglob('*.mrxs') if not x.name.startswith('.')]
    
    
    save_dir = Path.home() / 'workspace/localization' / f'TILS_candidates_{roi_size_ori}'
    save_dir.mkdir(exist_ok = True, parents = True)
    #save_dir = Path.home() / 'workspace/localization/TILS_candidates_small'
    #%%
    slides_dict = {}
    for fname in slides_fnames:
        fkey = f'{fname.parent.name}-{fname.stem}'
        assert not fkey in slides_dict
        slides_dict[fkey] = fname
    
    
    colors = {'L':'r', 'E':'g'}
    for coords_fname in tqdm.tqdm(coords_files):
        
        fkey = f'{coords_fname.parent.name}-{coords_fname.stem}'
        
        if not fkey in slides_dict:
            continue
        
        coords = pd.read_csv(coords_fname) 
        
        if only_limphocytes:
            coords = coords[coords['0'] == 'L']
        
        slide_fname = slides_dict[fkey]
        
        reader =  OpenSlide(str(slide_fname))
        _size = reader.level_dimensions[0]
        objective = reader.properties['openslide.objective-power']
        
        if objective == '20':
            roi_size = roi_size_ori
        elif objective == '40':
            roi_size = roi_size_ori*2
        else:
            ValueError(objective)
        
        for _type, dat in coords.groupby('0'):
           
            xx = coords['1']
            yy = coords['2']
            
            grid_size = np.ceil([s / roi_size for s in _size]).astype(np.int)
            
            bin_x = np.floor(xx/roi_size).values.astype(np.int)
            bin_y = np.floor(yy/roi_size).values.astype(np.int)
            
            #I do not want to include the data from the borders since it is likely to be cropped
            
            good = (bin_x < grid_size[0] -1) & (bin_y < grid_size[1] -1)
            bin_x = bin_x[good]
            bin_y = bin_y[good]
            
            bin_ii = (bin_x + bin_y*grid_size[0]).astype(np.int)
            
            counts = np.bincount(bin_ii)
            
            if ignore_high_density:
                counts[counts > 100] = -1
            
            #select top_k from the top_n without replacement
            top_ii = np.argsort(counts)[-top_n:]
            
            top_ii = top_ii[counts[top_ii] > min_counts[_type]]
            if top_ii.size == 0:
                continue
            
            top_ii = list(top_ii)
            random.shuffle(top_ii)
            for ind in top_ii[:top_k]:
            
                corner_d =  ind % grid_size[0], ind // grid_size[0]
                corner = [int(x*roi_size) for x in corner_d]
                
                roi = reader.read_region(corner, 0, (roi_size, roi_size))
                roi = np.array(roi)[..., :-1]
                
                save_name = save_dir / f'{_type}_{coords_fname.stem}_ROI-({corner[0]},{corner[1]})-{roi_size}.png'
                cv2.imwrite(str(save_name), roi[..., ::-1])
        
     