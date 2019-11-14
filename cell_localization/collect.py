#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:47:09 2019

@author: avelinojaver
"""

import numpy as np
import tables

_filters = tables.Filters(
        complevel=5,
        complib='blosc:lz4',
        shuffle=True,
        fletcher32=True)

def save_data(save_name, src_files, images, centroids = None, masks = None, contours = None):
    
    max_length = max(len(x[1]) for x in src_files)
    src_files_rec = np.array(src_files,  [('file_id', np.int32), ('file', f'S{max_length}')])
    
    save_name.parent.mkdir(exist_ok=True, parents=True)
    
    
    
    with tables.File(str(save_name), 'w') as fid:
        
        fid.create_carray('/', 'images', obj = images, chunkshape = (1, *images[0].shape), filters = _filters)
        fid.create_table('/', 'src_files', obj = src_files_rec, filters = _filters)
        
        if masks is not None:
            fid.create_carray('/', 'masks', obj = masks, chunkshape = (1, *masks[0].shape), filters = _filters)
        
        if centroids is not None:
            dtypes_centroids = [('file_id', np.int32), ('nuclei_id', np.int32), ('cx', np.float32), ('cy', np.float32) ]
            if len(centroids[0]) == 5:
                dtypes_centroids += [('type_id', np.int32)]
            
            centroids_rec = np.array(centroids, dtype = dtypes_centroids)
            fid.create_table('/', 'localizations', obj = centroids_rec, filters = _filters)
        
        if contours is not None:
            contours_rec = np.array(contours, dtype = [('file_id', np.int32), ('nuclei_id', np.int32), ('x', np.float32), ('y', np.float32) ])
            fid.create_table('/', 'contours', obj = contours_rec, filters = _filters)

def save_data_single(save_name, image, centroids = None, mask = None, contours = None):
    
    
    save_name.parent.mkdir(exist_ok=True, parents=True)
    
    with tables.File(str(save_name), 'w') as fid:
        
        fid.create_carray('/', 'img', obj = image, filters = _filters)
        
        if mask is not None:
            fid.create_carray('/', 'mask', obj = mask, filters = _filters)
        
        if centroids is not None:
            dtypes_centroids = [('nuclei_id', np.int32), ('cx', np.float32), ('cy', np.float32) ]
            if len(centroids[0]) == len(dtypes_centroids) + 1:
                dtypes_centroids += [('type_id', np.int32)]
            
            centroids_rec = np.array(centroids, dtype = dtypes_centroids)
            fid.create_table('/', 'coords', obj = centroids_rec, filters = _filters)
        
        if contours is not None:
            contours_rec = np.array(contours, dtype = [('nuclei_id', np.int32), ('x', np.float32), ('y', np.float32) ])
            fid.create_table('/', 'contours', obj = contours_rec, filters = _filters)