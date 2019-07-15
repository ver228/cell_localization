#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:34:04 2019

@author: avelinojaver
"""
from pathlib import Path 
import pandas as pd
import tqdm

import matplotlib.pylab as plt
from openslide import OpenSlide

if __name__ == '__main__':
    slides_dir = Path().home() / 'projects/bladder_cancer_tils/raw/'
    
    
    bn = 'bladder-tiles-roi64-20x_unetv2_l1smooth_20190529_193223_adam_lr6.4e-05_wd0.0_batch64'
    coords_dir = Path.home() / 'workspace/localization/predictions/histology_detection' / bn
    coords_files = coords_dir.rglob('*.csv')
    
    
    slides_fnames = [x for x in slides_dir.rglob('*.svs') if not x.name.startswith('.')]
    slides_fnames += [x for x in slides_dir.rglob('*.mrxs') if not x.name.startswith('.')]
    
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
        
        slide_fname = slides_dict[fkey]
        
        
        #%%
        reader =  OpenSlide(str(slide_fname))
        max_level =  reader.level_count - 1
        
        _size = reader.level_dimensions[max_level]
        downsample = reader.level_downsamples[max_level]
        coords = pd.read_csv(coords_fname)
        #%%
        img = reader.read_region((0,0), max_level, _size)
        
#        _corner = (50000,35000)
#        img = reader.read_region(_corner, 0, (2048, 2048))
#        
        #%%
        plt.figure()
        plt.imshow(img)
        for _type, dat in coords.groupby('0'):
            c = colors[_type]
            x = dat['1'].values/downsample
            y = dat['2'].values/downsample
            
#            x = dat['1'].values - _corner[0]
#            y = dat['2'].values - _corner[1]
            
            
            plt.plot(x, y, '.')
          #%% 
        
        