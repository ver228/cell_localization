#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:21:30 2019

@author: avelinojaver
"""

from openslide import OpenSlide, lowlevel

from pathlib import Path
import numpy as np
import cv2
import tqdm
import random
import pandas as pd
from collections import Counter

def get_single_candidate(img_rgb, low_th = 70, high_th = 230, std_th=5):
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))    
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))    
    
    rr = np.std(img_rgb, axis=2)
    _, valid_mask_p = cv2.threshold(rr, std_th, 255, cv2.THRESH_BINARY)
    valid_mask_p = valid_mask_p.astype(np.uint8)
    
    
    dark_mask_p = np.any(img_rgb<low_th, axis=-1).astype(np.uint8)*255
    bright_mask_p = np.any(img_rgb>high_th, axis=-1).astype(np.uint8)*255
    
    
    dark_mask = cv2.erode(dark_mask_p, k1)
    dark_mask = cv2.dilate(dark_mask, k2)
    
    bright_mask = cv2.erode(bright_mask_p, k1)
    bright_mask = cv2.dilate(bright_mask, k2)
    
    valid_mask = cv2.bitwise_not(cv2.bitwise_or(dark_mask, bright_mask))
    valid_mask = cv2.bitwise_and(valid_mask, valid_mask_p)
    
    im2, cnts, hierarchy = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(cnts, key = cv2.contourArea)
    
    best_mask = np.zeros_like(valid_mask)
    cv2.drawContours(best_mask, [largest_cnt], 0, 255, -1)
    
    #valid_mask_p = cv2.bitwise_not(bright_mask)
    best_mask = cv2.bitwise_and(best_mask, valid_mask_p)
    return best_mask

if __name__ == '__main__':
    #%%
    #tissue microtome
    TMS_files = ['1047012Alast', '1048151Alast', '10014931Blast', '4Afirstx20']
    
    pred_dir = Path.home() / 'workspace/localization/predictions/histology_detection/'
    #pred_dir = '/Volumes/rescomp1/data/localization/predictions/histology_detection/bladder-cancer-tils-40x_unet_l1smooth_20190416_211621_adam_lr0.00064_wd0.0_batch64/HEs'
    pred_dir = Path(pred_dir)
    pred_files_d = {x.stem:x for x in pred_dir.rglob('*.csv')}
    
    #slides_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/raw'
    slides_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/raw/HEs'
    slides_dir = Path(slides_dir)
    
    slide_files = [x for x in slides_dir.rglob('*.svs') if not x.name.startswith('.')]
    slide_files += [x for x in slides_dir.rglob('*.mrxs') if not x.name.startswith('.')]
    
    
    save_dir = Path.home() / 'workspace/localization/data/histology_bladder/roi_samples'
    
    #%%
    low_th = 70
    high_th = 230
    samples_per_roi_b = 10
    
    
    #%%
    roi_sizes_d = {'20' : 512, '40' : 1024} 
    
    #%%
    for islide, slide_file in enumerate(tqdm.tqdm(slide_files)):
        if slide_file.stem in pred_files_d:
            
            pred_file = pred_files_d[slide_file.stem]
            coords = pd.read_csv(pred_file)
            coords.columns = ['label', 'x', 'y']
            
            samples_per_roi = samples_per_roi_b
        else:
            continue
            #coords = None
            #samples_per_roi = samples_per_roi_b*3
        
        is_TMS = slide_file.stem in TMS_files
        
        try:
            reader =  OpenSlide(str(slide_file))
        except lowlevel.OpenSlideUnsupportedFormatError:
            if slide_file.name.endswith('mrxs'):
                continue
            else:
                raise(slide_file)
            
        objective = reader.properties['openslide.objective-power']
        vendor = reader.properties['openslide.vendor']
        
        roi_size = roi_sizes_d[objective]
        
        level_n = min(reader.level_count - 1, 7)
        level_dims = reader.level_dimensions[level_n]
        downsample = reader.level_downsamples[level_n]
        corner = (0,0)
        
        
        img = reader.read_region(corner, level_n, level_dims)
        
        img = np.array(img)
        img_rgb = img[..., :-1]
        
        if not is_TMS:
            
            best_mask = get_single_candidate(img_rgb, low_th, high_th)
        else:
            k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))    
            k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))    
            bright_mask_p = np.any(img_rgb>high_th, axis=-1).astype(np.uint8)*255
            bright_mask = cv2.erode(bright_mask_p, k1)
            bright_mask = cv2.dilate(bright_mask, k2)
            
            best_mask = cv2.bitwise_not(bright_mask)
        #%%
        
        _factor = downsample/roi_size
        tiny_mask = cv2.resize(best_mask, dsize=(0,0), fx=_factor, fy=_factor)
        tiny_mask = tiny_mask==255
        
        if coords is not None:
        
            corners_s = []
            for lab in ['E', 'L']:
                cc = coords.loc[coords['label'] == lab, ['x', 'y']].values
                
                
                cc_ind = np.floor(cc/roi_size).astype(np.int)
                cc_ind = cc_ind[cc_ind[:, 0] < tiny_mask.shape[1]]
                cc_ind = cc_ind[cc_ind[:, 1] < tiny_mask.shape[0]]
                
                cc_ind = [tuple(x) for x in cc_ind]
                cc_counts = Counter(cc_ind)
                
                
                rows, cols, vals = map(np.array, zip(*[(c,r, n) for (c,r), n in cc_counts.items()]))
                
                lab_mask = np.zeros_like(tiny_mask, np.int)
                
                try:
                    lab_mask[cols, rows] = vals
                except:
                    import pdb
                    pdb.set_trace()
                val_mask = tiny_mask.copy()
                val_mask[(lab_mask<5) | (lab_mask > 1000)] = 0
                
                
                
                v_rows, v_cols = np.where(val_mask)
                corners = list(zip(v_cols*roi_size, v_rows*roi_size))
                
                ss = min(len(corners), samples_per_roi)
                
                corners_s += random.sample(corners, ss)
                
        else:
            v_rows, v_cols = np.where(tiny_mask)
            corners = list(zip(v_cols*roi_size, v_rows*roi_size))
            corners_s = random.sample(corners, samples_per_roi)
        
        #%%
        ss = save_dir / slide_file.parent.name
        for corner in corners_s:
            img_o = reader.read_region(corner, 0, (roi_size, roi_size))    
            save_name = ss / f'{slide_file.stem}_{vendor}-{objective}_C{corner[0]}-{corner[1]}_R{roi_size}.png'
            
            save_name.parent.mkdir(exist_ok = True, parents = True)
            img_o = np.array(img_o)[..., -2::-1]
            
            cv2.imwrite(str(save_name), img_o)
            
            
        reader.close()
        
        #%%
        
        