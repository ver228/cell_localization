#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:47:02 2019

@author: avelinojaver
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tables
#%%
if __name__ == '__main__':
    
    save_dir = Path.home() / 'workspace/localization/test_images/'
    save_dir.mkdir(parents = True, exist_ok = True)
    
    img_size = (128, 128)
    n_samples = 50
    
    
    
    for nn in range(5):
        save_name = save_dir / f'{nn}.hdf5'
        
        x_coords = np.random.randint(0, img_size[0], n_samples)
        y_coords = np.random.randint(0, img_size[1], n_samples)
        
        coords = set([(x,y) for x,y in zip(x_coords, y_coords)])
        coords = np.array(list(coords))
        coords = pd.DataFrame({'type_id':1, 'cx':coords[:,0], 'cy':coords[:,1]})
        coords = coords.to_records(index=False)
        
        img = np.zeros(img_size, dtype=np.uint8)
        img[coords['cy'], coords['cx']] = 255
        
        with tables.File(str(save_name), 'w') as fid:
            
            fid.create_carray('/', 'img', obj = img)
            
            fid.create_table('/', 'coords', obj = coords)