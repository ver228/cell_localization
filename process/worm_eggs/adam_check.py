#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:48:55 2019

@author: avelinojaver
"""

from pathlib import Path
import pandas as pd
from collections import defaultdict
import matplotlib.pylab as plt
from scipy.stats import spearmanr, pearsonr
from scipy.stats import ttest_ind
import numpy as np

def df2counts(x):
    x = x['frame_number'].value_counts()
    x = {i:c for i,c in zip(x.index, x.values)}
    return x

if __name__ == '__main__':
    targets_dir = Path.home() / 'workspace/WormData/Adam_eggs/'
    
    #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
    bn = 'worm-eggs-adam+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190803_225943_adam_lr0.000128_wd0.0_batch64'
    preds_dir = Path.home() / 'workspace/localization/predictions/worm_eggs/' / 'Adam_eggs' / bn
    
    fnames = targets_dir.rglob('*.csv')
    fnames = [x for x in fnames if not x.name.startswith('.')]
    targets = {}
    for fname in fnames:
        df = pd.read_csv(fname)
        bn = fname.stem.rpartition('_')[0]
        targets[bn] = df
        
    
    
    #%%
    #valid img I do not have a good way of finding out if a frame is zero because there are no eggs or because it was not copied...
    #targets_dir
    images_fnames = [x for x in targets_dir.rglob('*.png') if not x.name.startswith('.')]
    valid_frames = defaultdict(list)
    for fname in images_fnames:
        base, _, frame = fname.stem.partition('_frame-')
        valid_frames[base].append(int(frame))
    #%%
    pred_fnames = preds_dir.rglob('*.csv')
    preds_dict = {}
    for fname in pred_fnames:
        df = pd.read_csv(fname)
        
        bn = fname.stem.rpartition('_')[0]
        preds_dict[bn] = df
       
    annotated_preds = {k:v for k,v in preds_dict.items() if k in targets}
    #%%
    #there seems to be something weird in the videos that do not have the 5 frames..
    unannotated_preds = {}
    for k,v in preds_dict.items():
        if not k in targets:
            ff = v['frame_number'].unique()
            if len(ff) >= 5:
                unannotated_preds[k] = v
    
    #%%
    weird_files = []
    
    
    
    res = defaultdict(list)
    for k, preds in annotated_preds.items():
        tt = df2counts(targets[k])
        #the annotations only add the new frames, so i need the cumsum
        true_counts = {}
        cur = 0
        for i in range(5):
            if i in tt:
                cur += tt[i]
            true_counts[i] = cur
        
        #top = preds[preds['frame_number']>0].max()
        #rel = preds['score_abs']/top['score_abs']
        #preds = preds[rel>0.2]
        #print(preds['score_abs'].min())
        
        pred_counts = df2counts(preds)
        
        for frame in sorted(valid_frames[k]):
            ct = true_counts[frame] if frame in true_counts else 0
            cp = pred_counts[frame] if frame in pred_counts else 0
            
            if ct > 0 and cp == 0:
                weird_files.append((k, frame))
            
            res[frame].append((ct, cp))
        
        
    #%%
    for k in range(5):
        true, pred = zip(*res[k])
        
        plt.figure()
        plt.plot(true, pred, '.')
    
        plt.xlabel('True')
        plt.ylabel('Prediction')
        
        plt.title(f'Frame = {k}')
        
        
        print(f'Frame = {k}')
        spearman_coeff, spearman_pval = spearmanr(true, pred)
        pearson_coeff, pearson_pval = pearsonr(true, pred)
        
        
        print(f'Spearman Coeff = {spearman_coeff}')
        print(f'Pearson Coeff = {pearson_coeff}')
        
    #%%
    res = []
    for bn, df in unannotated_preds.items():
        counts = df2counts(df)
        if 'No_Compound' in bn:
            strain, n_worms = bn.split('_')[:2]
            drug = 'No_Compound' 
            concentration  = 0
        else:
            strain, n_worms, drug, concentration = bn.split('_')[:4]
        
        n_worms = int(n_worms[5:])
        concentration = float(concentration)
        
        res += [(bn, strain, n_worms, drug, concentration, k, v) for k,v in counts.items()]
    
    
    res_all = pd.DataFrame(res, columns = ['basename', 'strain', 'n_worms', 'drug', 'concentration', 'frame', 'counts'])
    #%%
    for  frame2check in range(5):
        is_ctr = (res_all['drug'] == 'DMSO') 
        ctr = res_all[is_ctr]
        res = res_all[~is_ctr]
        
        res = res[(res['frame'] == frame2check)]
        ctr = ctr[(ctr['frame'] == frame2check)]
        
        pvals = []
        
        b = ctr['counts'].values
        for key, dat in res.groupby(['drug', 'concentration']):
            if len(dat) < 3:
                continue
            
            a = dat['counts'].values
            t, prob = ttest_ind(a, b)
            
            
            pvals.append((key, prob, np.mean(a)))
        pvals = sorted(pvals, key = lambda x : x[1])
        
        
        #bonferroni correction
        N = len(pvals)
        pvals_corr = [(key, pval*N, m) for key, pval, m in pvals]
        
        
        print(f'Frame = {frame2check}')
        for k, pval, m in pvals_corr:
            if pval < 0.1:
                dd = f'{k[0]}_{int(k[1]):03} | {pval:.2e} | {m:.2f}'
                print(dd)
    
    #%%
    all_diff = []
    
    cols = ['basename', 'strain', 'n_worms', 'drug', 'concentration', 'diff_counts']
    for bn, dat in res_all.groupby('basename'):
        dd = dat.copy()
        dd.index = dat['frame']
        
        try:
            ini = dd.loc[0]
            fin = dd.loc[4]
        except KeyError:
            continue
        
        count_diff = fin['counts'] - ini['counts']
        
        res = *[ini[x] for x in cols[:-1]], count_diff
        all_diff.append(res)
        
    all_diff = pd.DataFrame(all_diff, columns = cols)
    
    #%%
    is_ctr = (all_diff['drug'] == 'DMSO') 
    ctr = all_diff[is_ctr]
    res = all_diff[~is_ctr]
    
    
    pvals = []
    
    b = ctr['diff_counts'].values
    for key, dat in res.groupby(['drug', 'concentration']):
        if len(dat) < 3:
            continue
        
       
        a = dat['diff_counts'].values.copy()
        a[a<0] = 0
        t, prob = ttest_ind(a, b)
        
        
        pvals.append((key, prob, np.mean(a)))
    pvals = sorted(pvals, key = lambda x : x[0])
    
    
    #bonferroni correction
    N = len(pvals)
    pvals_corr = [(key, pval*N, m) for key, pval, m in pvals]
    
    print('Drug | p-val | Egg laying rate')
    for k, pval, m in sorted(pvals_corr, key=lambda x : x[1]):
        if pval < 0.1:
            dd = f'{k[0]}_{int(k[1]):03} | {pval:.2e} | {m:.2f}'
            print(dd)
    
    #%% plot control positions
    
    import cv2
    
    fnames_dict = {x.stem : x for x in images_fnames}
    
    #%%
    dat2check = res_all[res_all['drug'] == 'CSCD702471']
    for bn, _ in dat2check.groupby('basename'):
        
        fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
        for iframe, frame in enumerate([0, 4]):
          
            k = f"{bn}_frame-{frame}"
            try:
                fname = fnames_dict[k]
            except KeyError:
                continue
            
            img = cv2.imread(str(fname), -1)
            
            df = unannotated_preds[bn]
            
            coords = df.loc[df['frame_number'] == frame, ['x', 'y']]
            
            axs[iframe].imshow(img, cmap = 'gray')
            axs[iframe].plot(coords['x'], coords['y'], '.r')
    #%%
    cc = []
    for k, df in unannotated_preds.items():
        ff = df['frame_number'].unique()
        cc.append(len(ff))
        #print(min(df['score_abs']))
    
    
    