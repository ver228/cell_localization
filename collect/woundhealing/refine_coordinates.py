#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:03:09 2019

@author: avelinojaver
"""

import cv2
import pandas as pd
from pathlib import Path
import json
import numpy as np
import matplotlib.pylab as plt

from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

#%%

def correct_coords(img_, coords_, min_distance = 3, max_dist = 5):
    
    peaks = peak_local_max(img_, min_distance = min_distance)
    peaks = peaks[:, ::-1]
    
    
    #remove `peaks` that is not close by to any `coord` by at most `max_dist`
    D = cdist(coords_, peaks)
    good = (D <= max_dist).any(axis=0)
    D = D[:, good]
    valid_peaks = peaks[good]
    
    #find the closest peaks
    closest_indexes = np.argmin(D, axis=1)
    
    #we will consider as an easy assigment if the closest peak is assigned to only one coord
    u_indexes = np.unique(closest_indexes)
    counts = np.bincount(closest_indexes)[u_indexes]
    easy_assigments = u_indexes[counts == 1]
    valid_pairs = [(ii, x) for ii, x in enumerate(closest_indexes) if x in easy_assigments]
    
    easy_rows, easy_cols = map(np.array, zip(*valid_pairs))
    
    easy_cost = D[easy_rows, easy_cols]
    good = easy_cost<max_dist
    easy_rows = easy_rows[good]
    easy_cols = easy_cols[good]
    
    assert (D[easy_rows, easy_cols] <= max_dist).all()
    
    #now hard assigments are if a peak is assigned to more than one peak
    ambigous_rows = np.ones(D.shape[0], np.bool)
    ambigous_rows[easy_rows] = False
    ambigous_rows, = np.where(ambigous_rows)
    
    
    ambigous_cols = np.ones(D.shape[1], np.bool)
    ambigous_cols[easy_cols] = False
    ambigous_cols, = np.where(ambigous_cols)
    
    D_r = D[ambigous_rows][:, ambigous_cols]
    good = (D_r <= max_dist).any(axis=0)
    D_r = D_r[:, good]
    ambigous_cols = ambigous_cols[good]
    
    #for this one we use the hungarian algorithm for the assigment. This assigment is to slow over the whole matrix
    ri, ci = linear_sum_assignment(D_r)
    
    hard_rows, hard_cols = ambigous_rows[ri], ambigous_cols[ci]
    
    assert (D_r[ri, ci] == D[hard_rows, hard_cols]).all()
    
    
    hard_cost = D[hard_rows, hard_cols]
    good = hard_cost<max_dist
    hard_rows = hard_rows[good]
    hard_cols = hard_cols[good]
    
    #let's combine both and assign the corresponding peak
    rows = np.concatenate((easy_rows, hard_rows))
    cols = np.concatenate((easy_cols, hard_cols))
    
    new_coords = coords_.copy()
    new_coords[rows] = valid_peaks[cols] #coords that do not satisfy the close peak condition will not be changed
    
    return new_coords

     
        
#%%
        
        
#%%

if __name__ == '__main__':
    _debug = False
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/'
    root_dir = Path(root_dir)
    
    #annotations_file = root_dir / 'raw'/ 'dot_annotations.csv'
    annotations_file = root_dir / 'raw'/ 'annotations_v2.csv'
    
    save_dir = root_dir / 'data'
    
    (save_dir / 'train').mkdir(exist_ok=True, parents=True)
    (save_dir / 'test').mkdir(exist_ok=True, parents=True)
    
    df = pd.read_csv(str(annotations_file))
    
    for bn, dat in df.groupby('filename'):
            
        fname = annotations_file.parent / bn
        img = cv2.imread(str(fname), -1)
        
        #if not int(fname.stem) in [2, 3, 4, 5]:
        #    continue
        
        if int(fname.stem) in [12, 13, 14, 15, 16, 17]:
            continue
        
        
        fname_nuclei = fname.parents[1] / 'demixed' / ('N_' + fname.name)
        img_nuclei = cv2.imread(str(fname_nuclei), -1) if fname_nuclei.exists() else None
        
        fname_membrane = fname.parents[1] / 'demixed' / ('M_' + fname.name)
        img_membrane = cv2.imread(str(fname_membrane), -1) if fname_nuclei.exists() else None
        
        #print(img.min(), img.max())
        
        dd = [json.loads(x) for x in dat['region_shape_attributes']]
        coords = np.array([(x['cx'], x['cy']) for x in dd if 'cx' in x])
        
        #if img_nuclei is None:
        #    continue
        
        #new_coords = correct_coords(img_nuclei, coords)
        
        img_b = cv2.blur(img, (5,5))
        new_coords = correct_coords(img_b, coords)


        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.plot(coords[..., 0], coords[..., 1], 'xr')
        plt.plot(new_coords[..., 0], new_coords[..., 1], '.g')
        plt.title(fname.stem)

        #5.tif 4.tif 3.tif? 2.tif?
        #13.tif 14.tif 15.tif 16.tif 17.tif
        #%%
#        if img_membrane is not None or img_nuclei is not None:
#            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
#            axs[0].imshow(img, cmap='gray')
#            axs[0].plot(coords[..., 0], coords[..., 1], '.r')
#            axs[0].plot(peaks[..., 1], peaks[..., 0], '.g')
#            
#            if img_nuclei is not None:
#                axs[1].imshow(img_nuclei, cmap='gray')
#                axs[1].plot(coords[..., 0], coords[..., 1], '.r')
#            
#            if img_membrane is not None:
#                axs[2].imshow(img_membrane, cmap='gray')
#                axs[2].plot(coords[..., 0], coords[..., 1], '.r')
#            
#            plt.suptitle(fname.stem)
                  #%%               
        
        