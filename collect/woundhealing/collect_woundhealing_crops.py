#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:03:09 2019

@author: avelinojaver
"""

from collections import defaultdict
import cv2
import pandas as pd
from pathlib import Path
import json
import numpy as np
import tables
import math
import random

#%%


filters = tables.Filters(complevel=0, 
                          complib='blosc', 
                          shuffle=True, 
                          bitshuffle=True, 
                          fletcher32=True
                          )

if __name__ == '__main__':
    _debug = False
    
    validation_frac = 0.05
    test_frac = 0.05
    
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/'
    root_dir = Path(root_dir)
    
    
    for set_type in ['nuclei', 'mix']:
    
        imgs_dir = root_dir / 'raw' / set_type
        annotations_file = root_dir / 'annotations_v2' / f'{set_type}_via_region_data.csv'
        
        save_dir = root_dir / 'data4train' / set_type
        (save_dir / 'train').mkdir(parents=True)
        (save_dir / 'validation').mkdir(parents=True)
        (save_dir / 'test').mkdir(parents=True)
        
        
        
        #%%
        df = pd.read_csv(str(annotations_file))
        files2read = defaultdict(list)
        for ori_fname, dat in df.groupby('filename'):
            remain, _, src_img = ori_fname.partition('_')
            bn = src_img[:-4]
            src_img = bn + '.tif'
            
            rr = remain.split('-')[1:]
            roi_size, corner_x, corner_y = int(rr[0]), int(rr[1][1:]), int(rr[2][:-1])
            
            (xr,xl), (yr,yl) = (corner_x,corner_x + roi_size), (corner_y, corner_y + roi_size)
            
            roi_info = (xr,xl), (yr,yl)
            
            
            dd = [json.loads(x) for x in dat['region_shape_attributes']]
            coords = np.array([(x['cx'], x['cy']) for x in dd if 'cx' in x])
            
            if len(coords):
                files2read[src_img].append([bn, roi_info, coords])
        #%%
        src_files = list(files2read.keys())
        n_files = len(src_files)
        
        n_validation = math.ceil(n_files*test_frac)
        n_test = math.ceil(n_files*validation_frac)
        n_train = n_files - n_validation - n_test
        
        labels = ['train']*n_train + ['validation']*n_validation + ['test']*n_test
        random.shuffle(labels)
        
        
        for lab, src in zip(labels, src_files):
            
            fname = imgs_dir / src
            img = cv2.imread(str(fname), -1)
            
            for bn, roi_info, coords in files2read[src]:
                (xr,xl), (yr,yl) = roi_info
                roi = img[xr:xl, yr:yl]
                
                dat2save = pd.DataFrame({'type_id':1, 
                                                'cx':coords[:, 0], 
                                                'cy':coords[:, 1]})
                dat2save_rec = dat2save.to_records(index=False)       
                
                save_name = save_dir / lab / (bn + '.hdf5')
                with tables.File(str(save_name), 'w') as fid:
                    fid.create_carray('/', 'img', obj = roi, filters  = filters)
                    fid.create_table('/', 'coords', obj = dat2save_rec)
                        
                        