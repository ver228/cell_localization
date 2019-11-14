#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:58:48 2019

@author: avelinojaver
"""

from pathlib import Path
import pickle

if __name__ == '__main__':
    root_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/woundhealing-v2-mix/different_losses/'
    root_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/woundhealing-v2-mix/different_losses_complete/'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/woundhealing-v2-nuclei/different_losses/'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-F0.5-merged/'
    
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/eosinophils/20x/eosinophils-20x/different_losses'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/eosinophils/20x/eosinophils-20x/different_models'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/lymphocytes/20x/'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/lymphocytes_bad_downsamping/20x/'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/lymphocytes/40x/'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/eggs/worm-eggs-adam'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/eggs/worm-eggs-adam-masks'
    
    score_file = root_dir / 'scores.p'
    
    with open(score_file, 'rb') as fid:
        results = pickle.load(fid)
    
    #%%
    
    best_scores = []
    for bn, res in results.items():
        parts = bn.split('_')
        loss_name = parts[2]
        model_name = parts[1]
        ss = parts[0].rpartition('roi')[-1]
        
        
        th_str = f"th_{res['best_threshold_type']}={res['best_threshold_value']:.3}"
        #th_str = f"th_{res['best_threshold']}"
        
        F1_str = f"F1={res['test_scores']['F1']:.3}"
        
        dd = (F1_str, parts[0], loss_name, model_name, res['epoch'] , th_str, bn)
        best_scores.append(dd)
        
    best_scores = sorted(best_scores)
    for dd in best_scores:
#        if dd[-3] < 0.79:
#            continue
        
        print(dd[:-1])
        
        
    #%%
    #bn = 'eosinophils-20x+Feosinophils+roi48_unet-simple_l2-G2.5_20190727_020148_adam_lr0.000256_wd0.0_batch256'
    #res = results[bn]