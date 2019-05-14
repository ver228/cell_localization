#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:59:05 2019

@author: avelinojaver
"""

from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == '__main__':
    data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs/'
    fnames = data_root_dir.rglob('*.hdf5')
    
    set_dicts = {}
    for fname in fnames:
        dd = str(fname).replace(str(data_root_dir), '')
        set_type = dd.split('/')[1]
        
        bn = fname.stem.rpartition('_')[0].partition('_')[-1]
        set_dicts[bn] = set_type
    #%%
    bn = 'eggs-int_unet_hard-neg-freq1_l1smooth_20190513_081902_adam_lr0.000128_batch128'
    preds_dir = Path.home() / 'workspace/localization/predictions/worm_eggs/' / bn
    
    fnames = preds_dir.rglob('*.csv')
    
    preds_dict = {}
    for fname in fnames:
        df = pd.read_csv(fname)
        
        bn = fname.stem.rpartition('_')[0]
        preds_dict[bn] = df
    
    #%%
    targets_dir = Path.home() / 'workspace/WormData/screenings/Drug_Screening/MaskedVideos/'
    fnames = targets_dir.rglob('*.csv')
    targts_dir = {}
    for fname in fnames:
        df = pd.read_csv(fname)
        
        bn = fname.stem.rpartition('_')[0]
        targts_dir[bn] = df
    #%%
    counts_l = []
    for bn, set_type in set_dicts.items():
        preds_df = preds_dict[bn]
        target_df = targts_dir[bn]
        
        preds_counts = np.bincount(preds_df['frame_number'].values)
        target_counts = np.bincount(target_df['frame_number'].values)
        for frame_number, (p_c, t_c) in enumerate(zip(preds_counts, target_counts)):
            counts_l.append((set_type, bn, frame_number, p_c, t_c))
    
    counts_df = pd.DataFrame(counts_l, columns = ['set_type', 'basename', 'frame_number', 'pred_counts', 'target_counts'])
        
    #%%
    
    df_train = counts_df[counts_df['set_type'] == 'train']
    df_train = counts_df[counts_df['frame_number'] > 0]
    
    df_test = counts_df[counts_df['set_type'] != 'train']
    #%%
    
    np.corrcoef(df_train['pred_counts'], df_train['target_counts'])
    np.corrcoef(df_test['pred_counts'], df_test['target_counts'])