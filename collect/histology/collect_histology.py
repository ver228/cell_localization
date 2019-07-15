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
from sklearn.cluster import KMeans
from openslide import OpenSlide

#%%

if __name__ == '__main__':
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/')
    
    raw_data_dir = root_dir / 'raw'/ '40x'
    save_dir = root_dir / 'rois' / '40x'
    
    save_dir.mkdir(exist_ok=True, parents=True)
    
    fnames = list(raw_data_dir.glob('*.svs'))
    
    #%%
    
    test_frac = 0.1
    max_roi_size = 2000
    border_offset = 20
    min_roi_size = 200
    
    types_ids = {'Eosinophils' : 2, 'Lymphocytes' : 1}
    
    
    nn = math.ceil(test_frac*len(fnames))
    is_test_files = nn*[True] + [False]*(len(fnames)-nn)
    random.shuffle(is_test_files)
    
    for is_test_file, fname in tqdm.tqdm(zip(is_test_files, fnames)):
        exp_id = fname.stem
        annotations_file = fname.parent / (exp_id + '_annotation.json')
        
        
        with open(str(annotations_file), 'r') as fid:
            dd = json.load(fid)
        
        #%%
        annotations = []
        for lab_group in dd['layers']:
            lab_name = lab_group['name']
            type_id = types_ids[lab_name]
            #lab_name = np.array(lab_name)
            
            dat = [(lab_name, type_id, x['radius'], x['center']['x'], x['center']['y']) for x in lab_group['items']]
            annotations += dat
        annotations = pd.DataFrame(annotations, columns = ['type', 'type_id', 'radius', 'cx', 'cy'])
        
        if len(annotations) == 0:
            print(f'BAD : {exp_id}')
            continue
        #%%
        
        def _max_cluster_roi_size(km, coords):
            roi_sizes = []
            for nn in range(km.n_clusters):
                good = km.labels_ == nn
                cluster_data = coords[good]
                rr = max(cluster_data.max(axis=0) - cluster_data.min(axis=0))
                roi_sizes.append(rr)
                
            return max(roi_sizes)
        
        reader = OpenSlide(str(fname))
        
        slide_size = np.array(reader.dimensions)
        
        data = annotations[[ 'cx', 'cy']].values
        
        for n_clusters in range(1, 200):
            km = KMeans(n_clusters=n_clusters).fit(data)
            max_cluster_roi_size = _max_cluster_roi_size(km, data)
            if max_cluster_roi_size < max_roi_size:
                break
            
        #%%
        
        clusters_ = []
        for nn in range(km.n_clusters):
            good = km.labels_ == nn
            cluster_data = data[good]
            #%%
            top, bot = cluster_data.max(axis=0),  cluster_data.min(axis=0)
            
            bot = (bot - border_offset).astype(np.int)
            top = (top + border_offset).astype(np.int)
            
            #fix region to force a minimum roi size
            roi_size = top - bot
            
            for ii in range(2):
                if (roi_size[ii] < min_roi_size):
                    pad = math.ceil((min_roi_size - roi_size[ii])/2)
                    bot[ii] -= pad
                    top[ii] += pad
            
            roi_size = top - bot
            
            ll = tuple(bot), tuple(roi_size)
            
            
            valid = (annotations['cx'] > bot[0]) & (annotations['cy'] > bot[1]) & \
            (annotations['cx'] < top[0]) & (annotations['cy'] < top[1]) 
            
            valid_coords = annotations[valid].copy()
            valid_coords['cx'] -= bot[0]
            valid_coords['cy'] -= bot[1]
            
            clusters_.append((ll, valid_coords))
            
        
        #    num_workers = 8
#    batch_size = 16
#    loader = DataLoader(gen, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#    
#    
#    bot_x, top_x = 1e10, -1
#    bot_y, top_y = 1e10, -1
#    
#    for ii, (X,Y) in enumerate(tqdm.tqdm(loader)):
#        bot_x = min(bot_x, X.min())
#        top_x = max(top_x, X.max())
#        
#        bot_y = min(bot_x, X.min())
#        top_y = max(top_x, X.max())
#        
#    
#    print(bot_x, top_y)
#    print(bot_y, top_y)
        
        #tables cannot save 'O' as data type (default in strings)
        valid_dtypes = []
        for c in annotations:
            col_data = annotations[c]
            if col_data.dtype == 'O':
                s_size = col_data.str.len().max()
                dd = (c, f'<S{s_size}')
                
            else:
                dd = (c, col_data.dtype)
            valid_dtypes.append(dd)
        
        
        tot = len(clusters_)
        nn = math.floor(tot*test_frac)
        is_test_rois = (tot-nn)*[False] + nn*[True]
        random.shuffle(is_test_rois)
        
        for  is_test_roi, ((corner, roi_size), roi_data) in zip(is_test_rois, clusters_):
            
            is_test = is_test_roi | is_test_file
            
            roi = reader.read_region(corner, 0, roi_size)
            roi = np.array(roi)[..., :-1]
            
            coords = roi_data.to_records(index=False)
            coords = coords.astype(valid_dtypes)
            
            if is_test:
                ss = save_dir / 'test'
            else:
                ss = save_dir / 'train'
            
            save_name = ss / exp_id / f'{exp_id}_({corner[0]}-{corner[1]})_({roi_size[0]}-{roi_size[1]}).hdf5'
            save_name.parent.mkdir(exist_ok=True, parents=True)
            with tables.File(str(save_name), 'w') as fid:
                
                fid.create_carray('/', 'img', obj = roi)
                fid.create_table('/', 'coords', obj = coords)
            
            
    #import matplotlib.pylab as plt
    #        plt.figure()
    #        plt.imshow(roi)
    #        plt.plot(roi_data['cx'], roi_data['cy'], '.r')
            
            
            #plt.plot(row['cx'] - corner[0], row['cy'] - corner[1], 'or')
        