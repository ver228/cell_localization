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
    data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs_first/'
    targets_dir = Path.home() / 'workspace/WormData/screenings/Drug_Screening/MaskedVideos/'
    #bn = 'eggs-int_unet_hard-neg-freq1_l1smooth_20190513_081902_adam_lr0.000128_batch128'
    bn = 'eggsadamI-roi96_unetv2b-bn_hard-neg-freq10_maxlikelihoodpooled_20190619_081454_adam_lr3.2e-05_wd0.0_batch64'
    preds_dir = Path.home() / 'workspace/localization/predictions/worm_eggs/' / bn
    
    
    
    fnames = data_root_dir.rglob('*.hdf5')
    
    
    #get if it is train, test or validation
    set_dicts = {}
    for fname in fnames:
        dd = str(fname).replace(str(data_root_dir), '')
        set_type = dd.split('/')[1]
        
        bn = fname.stem.rpartition('_')[0].partition('_')[-1]
        set_dicts[bn] = set_type
    #%%
    fnames = preds_dir.rglob('*.csv')
    
    preds_dict = {}
    for fname in fnames:
        df = pd.read_csv(fname)
        
        bn = fname.stem.rpartition('_')[0]
        preds_dict[bn] = df
    
    #%%
    
    fnames = targets_dir.rglob('*.csv')
    fnames = [x for x in fnames if not x.name.startswith('.')]
    targets = {}
    for fname in fnames:
        df = pd.read_csv(fname)
        
        bn = fname.stem.rpartition('_')[0]
        targets[bn] = df
    #%%
    counts_l = []
    
    
    
    def get_counts(x, max_frame = 5):
        if len(x) > 0:
            return np.bincount(x, minlength = max_frame)
        else:
            return np.zeros((max_frame) , np.int)
    
    for bn, set_type in set_dicts.items():
        if not (bn in preds_dict and bn in targets):
            continue
        
        strain_name, _, drug_name, drug_concentration = bn.split('_')[:4]
        
        preds_df = preds_dict[bn]
        target_df = targets[bn]
        
        preds_counts = get_counts(preds_df['frame_number'].values)
        
        
        target_counts = get_counts(target_df['frame_number'].values)
        
        for frame_number, (p_c, t_c) in enumerate(zip(preds_counts, target_counts)):
            counts_l.append((set_type, bn, strain_name, drug_name, drug_concentration, frame_number, p_c, t_c))
    
    counts_df = pd.DataFrame(counts_l, columns = ['set_type', 'basename', 'strain_name', 'drug_name', 'drug_concentration', 'frame_number', 'pred_counts', 'target_counts'])
        
    #%%
    
    import matplotlib.pylab as plt
    plt.figure()
    
    
    plt.plot(counts_df['pred_counts'], counts_df['target_counts'], '.')
    plt.ylabel('True Counts')
    plt.xlabel('Predicted Counts')
    
    cc = plt.xlim()
    plt.plot(cc, cc, ':k')
    #%%
    for strain_name, strain_data in counts_df.groupby('strain_name'):
            
        for drug_name, drug_df in strain_data.groupby('drug_name'):
            if drug_name == 'Blank':
                continue
            
            fig, axs = plt.subplots(3, 1, figsize = (5, 15))
            for drug_conc, df in drug_df.groupby('drug_concentration'):
            
                
                axs[0].plot(df['pred_counts'], df['target_counts'], 'o', label = drug_conc)
            
                axs[1].plot(df['frame_number'], df['pred_counts'], 'o')
                axs[2].plot(df['frame_number'], df['target_counts'], 'o')
            
            
            axs[0].set_ylabel('Target Counts')
            axs[0].set_xlabel('Predicted Counts')
            axs[0].plot(cc, cc, ':k')
            axs[0].legend()
            
            axs[1].set_xlabel('Frame Number')
            axs[1].set_xlabel('Predicted Counts')
            
            axs[2].set_ylabel('Frame Number')
            axs[2].set_xlabel('Target Counts')
            
            
            
            plt.suptitle((strain_name, drug_name))
            
    
    
    
    
    
    
    
    
    
    
    
    
    