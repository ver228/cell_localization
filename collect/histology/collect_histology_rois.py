#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:03:09 2019

@author: avelinojaver
"""

import pandas as pd
from pathlib import Path
import json
import numpy as np
import tables
import random
import math

import tqdm
import cv2


#%%

def get_valid_dtypes(annotations):
    valid_dtypes = []
    for c in annotations:
        col_data = annotations[c]
        if col_data.dtype == 'O':
            s_size = col_data.str.len().max()
            dd = (c, f'<S{s_size}')
            
        else:
            dd = (c, col_data.dtype)
        valid_dtypes.append(dd)
        
    return valid_dtypes

if __name__ == '__main__':
    annotations_file = Path('/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/raw/valid_TILs/Bladder_Tiles_Annotation_Study.csv')
    tiles_dir = annotations_file.parent
    
    
    save_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/full_tiles')
    save_dir.mkdir(exist_ok=True, parents=True)
    
    types_ids = {'Eosinophils' : 2, 'Lymphocytes' : 1}
    
    with open(annotations_file) as fid:
        ss = fid.read()
        ss = [x for x in ss.split('\n')[1:] if x]
        
    for line in ss:
        src, _, remain = line.partition(',')
        annotation_str, _, annotator = remain.rpartition(',')
        
        dd = json.loads(annotation_str[1:-1])
        annotations_l = []
        for lab_group in dd['annotation']['layers']:
            lab_name = lab_group['name']
            type_id = types_ids[lab_name]
            #lab_name = np.array(lab_name)
            
            dat = [(lab_name, type_id, x['radius'], x['center']['x'], x['center']['y']) for x in lab_group['items']]
            annotations_l += dat
        annotations_ori = pd.DataFrame(annotations_l, columns = ['type', 'type_id', 'radius', 'cx', 'cy'])
        
        
        src_file = tiles_dir / (src[1:-1] + '.png')
        img_ori = cv2.imread(str(src_file), -1)
        
        
        src_magification = [int(x.partition('-')[-1]) for x in src.split('_') if (x.startswith('mirax') or x.startswith('aperio'))][0]
        assert src_magification in [20, 40]
        
        for mm in [20, 40]:
            annotations = annotations_ori.copy()
            if src_magification == mm:
                save_name = save_dir / f'{mm}x' / f'{src_file.stem}.hdf5'
                img = img_ori
            else:
                save_name = save_dir / f'{mm}x' / f'{src_file.stem}_resized.hdf5'
                if mm == 40:
                    #%%
                    img = cv2.resize(img_ori, (0,0), fx=2., fy=2.)
                    annotations['cx'] *= 2
                    annotations['cy'] *= 2
                    annotations['radius'] *= 2
                elif mm == 20:
                    img = cv2.resize(img_ori, (0,0), fx=0.5, fy=0.5)
                    annotations['cx'] /= 2
                    annotations['cy'] /= 2
                    annotations['radius'] /= 2
                    
            #print(src_magification, mm, img.shape, img_ori.shape)
            print(img.min(), img.max())
            save_name.parent.mkdir(exist_ok=True, parents=True)
            with tables.File(str(save_name), 'w') as fid:
                
                fid.create_carray('/', 'img', obj = img)
                
                
                if len(annotations) > 0:
                    valid_dtypes = get_valid_dtypes(annotations)
                    coords = annotations.to_records(index=False)
                    valid_dtypes = coords = coords.astype(valid_dtypes)
                    
                    fid.create_table('/', 'coords', obj = coords)
            
#            import matplotlib.pylab as plt
#            plt.figure()
#            plt.imshow(img)
#            plt.plot(annotations['cx'], annotations['cy'], '.r')
#        break
            
            