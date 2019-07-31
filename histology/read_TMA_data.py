#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:40:48 2019

@author: avelinojaver
"""

from openslide import OpenSlide
import cv2

import pandas as pd
import json
from pathlib import Path
import numpy as np
import matplotlib.pylab as plt

from scipy.stats import spearmanr

if __name__ == '__main__':
    min_dist_grid = 1000
    sample_radii_range = (1280, 1920)
    downsample_level = 3 
    
    #slide_file = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/TMA_counts/2Afirst104834.svs'
    slide_file = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/TMA_counts/2Bfirst104868.svs'
    slide_file = Path(slide_file)
    
    
    bn = 'TH0.1_eosinophils-20x+Feosinophils+roi48_unet-simple_l2-G2.5_20190727_020148_adam_lr0.000256_wd0.0_batch256'
    global_th = 0.25
    
    #bn = 'TH0.0_eosinophils-20x+Feosinophilsonly+roi48+hard-neg-1_unet-simple_maxlikelihood_20190724_080046_adam_lr0.000256_wd0.0_batch256'
    #global_th = 0.004
    
#    bn = 'TH0.15_eosinophils-20x+Feosinophils+roi96_unet-simple_l2-G2.5_20190728_054209_adam_lr0.000256_wd0.0_batch256'
#    global_th = 0.15
    
    
    coords_fname = Path.home() / 'workspace/localization/predictions/histology_detection' / bn / 'TMA_counts' / (slide_file.stem + '.csv')
    coords = pd.read_csv(coords_fname)
    
    
    grid_file = slide_file.parent / (slide_file.stem + '_lines.json')
    assert grid_file.exists()
    with open(grid_file, 'r') as fid:
        grid_lines = json.load(fid)[::-1]
    
    landmark_file = slide_file.parent / (slide_file.stem + '_landmarks.json')
    assert landmark_file.exists()
    with open(landmark_file, 'r') as fid:
        landmarks = json.load(fid)
    
    GT_file = slide_file.parent / (slide_file.stem + '_GT-eosinophils.csv')
    GT_df = pd.read_csv(GT_file)
    
    
    
    #%%
    if 'Total Eo' in GT_df:
        GT = { (row['row'], row['col']) : row['Total Eo'] for _, row in GT_df.iterrows()}
    else:
        GT = { (row['row'], row['col']) : row['Tumour'] +row['Stroma'] for _, row in GT_df.iterrows()}
    
    #%% remove points that are too close. likely duplicated grids
    grid = []
    for _, line in grid_lines:
        line_new = []
        for p in  np.array(line):
            if len(line_new) > 0:
                d = line_new[-1] - p
                m = np.sqrt((d**2).sum())
                if m > 1000:
                    line_new.append(p)
                
            else:
                line_new.append(p)
        line_new = np.array(line_new)
        grid.append(line_new)
    grid = np.array(grid)
    
    
    #%%
    for lab, circ in landmarks:
        cm = np.mean(circ, axis=0)
        r2 = ((grid - cm[None, None])**2).sum(axis=-1)
        match = np.unravel_index(np.argmin(r2), r2.shape)
        k = tuple([x+1 for x in match]) 
        
        valid = (GT_df['row'] == match[0] + 1) & (GT_df['col'] == match[1] + 1)
        gt = GT_df.loc[valid, 'id'].iloc[0]
        
        if ':' in gt:
            gt = gt.partition(':')[-1]
        
        print(f'{gt} | {lab.title()}')
        
    #%%
    
    #%%
    print(max(GT.keys()), grid.shape)
    #%%
    reader =  OpenSlide(str(slide_file))
    
    slide_size = reader.level_dimensions[downsample_level]
    downsample = reader.level_downsamples[downsample_level]
    
    img = reader.read_region((0,0), downsample_level, slide_size)
    img = np.array(img)[..., :-1]
    #%%
    bw = (img<230).any(axis=2)
    bw = bw.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    
    _, cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [x for x in cnts if cv2.contourArea(x) > 500]
    
    circles = [cv2.minEnclosingCircle(x) for x in cnts]
    cms, radii = map(np.array, zip(*circles))
    r_med = np.median(radii)
    
    
    r_lim = [x/downsample for x in sample_radii_range]
    valid = (radii>r_lim[0]) & (radii<r_lim[1])
    cms, radii = cms[valid], radii[valid] 
    
    cnts_v = [x for x, v in zip(cnts, valid) if v]
    del cnts
    
    #%%
    grid_downsampled = grid / downsample
    
    grid_matches = []
    for cm in cms:
        r2 = ((grid_downsampled - cm[None, None])**2).sum(axis=-1)
        match = np.unravel_index(np.argmin(r2), r2.shape)
        
       
        grid_matches.append(match)
    
    
    #%%
    
    dat = coords[coords['score_abs'] > global_th]
    x_pred = dat['cx'].values/downsample
    y_pred = dat['cy'].values/downsample
    
    cnts_points = []
    for cnt in cnts_v:
        xmin = cnt[..., 0].min()
        xmax = cnt[..., 0].max()
        ymin = cnt[..., 1].min()
        ymax = cnt[..., 1].max()
        
        valid = (x_pred >= xmin) & (x_pred <= xmax) & (y_pred >= ymin) & (y_pred <= ymax)
        
        cnt_points = []
        for x, y in zip(x_pred[valid], y_pred[valid]):
            p = (x, y)
            d = cv2.pointPolygonTest(cnt, p, False)
            if d >= 0:
                cnt_points.append(p)
        cnts_points.append(cnt_points)
    
    
    fig, ax = plt.subplots()
    plt.imshow(img)
    plt.plot(x_pred, y_pred, '.b')
    #%%
    
    
    
    #%%
    fig, ax = plt.subplots()
    plt.imshow(img)
    plt.plot(grid_downsampled[..., 0], grid_downsampled[..., 1], '.g')
    
    
    for cm, r, (gi, gj) in zip(cms, radii, grid_matches):
        c = plt.Circle(cm, r, color='b', fill=False)
        ax.add_artist(c)
        
        match_c = grid_downsampled[gi, gj]
        plt.plot((match_c[0], cm[0]), (match_c[1], cm[1]), 'or')
    #%%
    
    
    res = []
    for match,v in zip(grid_matches, cnts_points):
        
        k = tuple([x+1 for x in match])
        
        pred = len(v)
        gt = GT[k]
        
        res.append((pred, gt))
        
        
    preds, true = map(np.array, zip(*res))
    
    good = ~np.isnan(true)
    
    rho, pval = spearmanr(preds[good], true[good])
    print(f'Spearman Rho {rho}')
    
    plt.figure()
    plt.plot(true, preds, '.')
    
    plt.ylabel('Predictions')
    plt.xlabel('GT')
    
    #%%
    #ind = np.argmax([len(x) for x in cnts_points])
    ind = np.nanargmax(np.abs(preds - true))
    cnt = cnts_v[ind].squeeze(1)*downsample
    points = np.array(cnts_points[ind])*downsample
    
    match = grid_matches[ind]
    k = tuple([x+1 for x in match])
    gt = GT[k]
    
    plt.figure()
    plt.plot(cnt[...,0], cnt[...,1])
    plt.plot(points[:, 0], points[:, 1], '.')
    
    corner = np.min(cnt, axis=0)
    roi_size = (np.max(cnt, axis=0) - corner)
    
    corner = np.floor(corner).astype(np.int)
    roi_size = np.ceil(roi_size).astype(np.int)
    
    
    roi = reader.read_region(corner, 0, roi_size)
    roi = np.array(roi)[..., :-1]
    
    plt.figure()
    plt.imshow(roi)
    plt.plot(cnt[...,0] - corner[0], cnt[...,1]-corner[1])
    plt.plot(points[...,0] - corner[0], points[...,1]-corner[1], '.')
    