#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:16:58 2019

@author: avelinojaver
"""
from pathlib import Path
import pandas as pd
import tqdm
import cv2
import random
import numpy as np
import os

if __name__ == '__main__':
    is_plot = False
    cuda_id = 0
    scale_int = (0, 4095)
    
    n_rois = 2
    roi_size = 192
    
    img_src_dir = Path.home() / 'workspace/localization/data/woundhealing/raw/'
    bn = 'woundhealing-all-roi48_unetv2-init-normal_l1smooth_20190531_143703_adam_lr0.000128_wd0.0_batch128'
    coords_dir = Path.home() / 'workspace/localization/data/woundhealing/location_predictions' / bn
    
    
    save_dir_root = Path('/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/rois2label')
    #save_dir_root = Path.home() / 'workspace/localization/data/woundhealing/rois2label'
    
    data2save = {}
    
    coord_files = coords_dir.rglob('*.csv')
    coord_files = list(coord_files)
    for coord_file in tqdm.tqdm(coord_files):
        df = pd.read_csv(coord_file)
        
        img_dir = str(coord_file.parent).replace(str(coords_dir), str(img_src_dir))
        img_dir = Path(img_dir)
        
        img_name = img_dir / coord_file.stem.rpartition('_')[0]
        img = cv2.imread(str(img_name), -1)
        
        save_dir = str(coord_file.parent).replace(str(coords_dir), str(save_dir_root))
        save_dir = Path(save_dir)
        save_dir.mkdir(parents = True, exist_ok = True)
        
        
        xseg_size = img.shape[1]//n_rois
        for iroi in range(n_rois):
            
            
            y_corner = random.randint(0, img.shape[0] - roi_size)
            x_corner = xseg_size*iroi + random.randint(0, xseg_size - roi_size)
            
            
            yl, yr = y_corner, y_corner+roi_size
            xl, xr = x_corner, x_corner+roi_size
            roi = img[yl:yr, xl:xr]
            
            good = (df['cy'] >= yl) & (df['cy'] < yr) & (df['cx'] >= xl) & (df['cx'] < xr)
            roi_coords = df[good]
            xx = roi_coords['cx'] - xl
            yy = roi_coords['cy'] - yl
            
            roi_l = np.log(roi.astype(np.float32))
            bot, top = np.min(roi_l), np.max(roi_l)
            roi_norm = ((roi_l - bot)/(top-bot)*255).astype(np.uint8)
            
            save_name = save_dir / f'ROI{iroi+1}-{roi_size}-({y_corner}-{x_corner})_{img_name.stem}.png'
            cv2.imwrite(str(save_name), roi_norm)
            
            
            _dname = save_name.parent
            if not _dname in data2save:
                data2save[_dname] = []
            
            statinfo = os.stat(save_name)
            
            fsize = statinfo.st_size
            fattrs = '{"caption":"","public_domain":"no","image_url":""}'
            region_count = len(xx)
            region_attributes = "{}"
            
            rows = []
            for region_id, (x,y) in enumerate(zip(xx, yy)):
                region_shape_attributes = "{" + f'"name":"point","cx":{x},"cy":{y}' + "}"
                row = (save_name.name, fsize, fattrs, region_count, region_id, region_shape_attributes, region_attributes)
                rows.append(row)
            data2save[_dname] += rows
            
    #%%
    columns = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes']
    for dname, rows in data2save.items():
        df = pd.DataFrame(rows, columns = columns)
        
        save_name = dname / f'{dname.name}_annotations.csv'
        df.to_csv(save_name, index=False)
        
    
    