#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:59:24 2019

@author: avelinojaver
"""

import pandas as pd
from pathlib import Path
import matplotlib.pylab as plt
import numpy as np
#from py4j.java_gateway import JavaGateway

from openslide import OpenSlide
#%%
if __name__ == '__main__':
    pred_dir = Path.home() / 'workspace/localization/predictions/histology_detection/bladder-cancer-tils_unet_l1smooth_20190406_000552_adam_lr0.00064_wd0.0_batch64'
    
    #slides_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/raw'
    slides_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/raw/'
    
    pred_dir = Path(pred_dir)
    slides_dir = Path(slides_dir)
    
    pred_files = pred_dir.glob('*.csv')
    slide_files_d = {x.stem:x for x in slides_dir.rglob('*.svs')}
    #slide_files_d = {x.stem:x for x in slides_dir.rglob('101TUR1-HE.m*')}
    
    
    
    #%%
    for pred_file in pred_files:
        coords = pd.read_csv(pred_file)
        coords.columns = ['label', 'x', 'y']
        
        
        slide_file = slide_files_d[pred_file.stem]
        reader =  OpenSlide(str(slide_file))
    
    
        level_n = reader.level_count - 1
        level_dims = reader.level_dimensions[level_n]
        downsample = reader.level_downsamples[level_n]
        corner = (0,0)
        
        if False:
            level_n = 0
            level_dims = reader.level_dimensions[level_n]
            downsample = reader.level_downsamples[level_n]
            
            corner = (18880,11200)
            #corner = (24000,11200)
             
            #corner = (550*32, 1180*32)
            #corner = (2850*32, 200*32)
            #corner = (1725*32, 80*32)
            
            level_dims = (3200, 3200)
        img = reader.read_region(corner, level_n, level_dims)
        
        
        coords_r = coords.copy()
        coords_r[['x', 'y']] /= downsample
        
        good = (coords_r['x'] >= corner[0]) & (coords_r['x'] <= corner[0] + level_dims[0])
        good &= (coords_r['y'] >= corner[1]) & (coords_r['y'] <= corner[1] + level_dims[1])
        coords_r = coords_r[good]
        
        
        img = np.array(img)
        #%%
        plt.figure()
        plt.imshow(img)
        
        
        colors = {'L' : 'r', 'E' : 'g'}
        for lab, dat in  coords_r.groupby('label'):
            x = dat['x'] - corner[0]
            y = dat['y'] - corner[1]
            plt.plot(x, y, '.', color  = colors[lab])
    