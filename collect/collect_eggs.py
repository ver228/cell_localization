#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 16:22:37 2019

@author: avelinojaver
"""

from pathlib import Path
import pandas as pd
import tables
import tqdm
import math
import random
import cv2
import numpy as np
from refine_coordinates import correct_coords
#%%
if __name__ == '__main__':
    root_dir = Path.home() / 'workspace/WormData/screenings/Drug_Screening/MaskedVideos/'
    root_dir = Path(root_dir)
    
    save_dir = Path.home() / 'workspace/localization/worm_eggs/'
    
    fnames = [x for x in root_dir.rglob('*.csv') if not x.name.startswith('.')]
    
    #annotations = pd.DataFrame(annotations, columns = ['type', 'type_id', 'radius', 'cx', 'cy'])
    
    frame2save = 0
    val_frac = 0.05
    test_frac = 0.03
    
    test_nn = math.ceil(len(fnames) * test_frac)
    val_nn = math.ceil(len(fnames) * test_frac)
    train_nn = len(fnames) - test_nn - val_nn
    
    set_types = ['test']*test_nn + ['train']*train_nn + ['val']*val_nn
    random.shuffle(set_types)
    
    for fname, set_type in tqdm.tqdm(zip(fnames, set_types)):
        mask_file = fname.parent / (fname.name.split('_eggs')[0] + '.hdf5')
        #%%
        with tables.File(str(mask_file), 'r') as fid:
            full_data = fid.get_node('/full_data')[:]
            
        df = pd.read_csv(str(fname))
        df = df[(df['frame_number'] == frame2save) & (df['group_name'] == '/full_data')]
        img = full_data[frame2save]
        
        if len(df) == 0:
            continue
        
        #refine coordinates, make sure the center of the coordinates corresponce, or at least is close by an egg center
        valid_mask = np.zeros_like(img)
        
        cols = df['x'].astype(np.int)
        rows = df['y'].astype(np.int)
        valid_mask[rows, cols] = 1
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15, 15))
        valid_mask = cv2.dilate(valid_mask, kernel) > 0
        
        img_peaks = ~img
        
        img_peaks -= img_peaks[valid_mask].min()
        img_peaks[~valid_mask] = 0
        img_peaks = cv2.blur(img_peaks, (5,5))
        
        cc = df[['x','y']].values
        new_coords = correct_coords(img_peaks, cc, min_distance = 3, max_dist = 5)
        
        
        coords = pd.DataFrame({'type_id':1, 'cx':new_coords[:,0], 'cy':new_coords[:,1]})

        coords = coords.to_records(index=False)
        
        ss = save_dir / set_type
        
        prefix = str(fname.parent).replace(str(root_dir), '')[1:]
        
        save_name = ss / prefix / f'EGGS_{fname.stem}.hdf5'
        save_name.parent.mkdir(exist_ok=True, parents=True)
        with tables.File(str(save_name), 'w') as fid:
            
            fid.create_carray('/', 'img', obj = img)
            
            fid.create_table('/', 'coords', obj = coords)
            
            #%%
#        import matplotlib.pylab as plt
#        
#        
#        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
#        
#        ax.imshow(img, cmap = 'gray')
#        ax.plot(coords['cx'], coords['cy'], '.r')
#        #ax.plot(new_coords[:,0], new_coords[:,1], '.g')
#        plt.show()
        #%%
        
            
        
        
        
        
    

