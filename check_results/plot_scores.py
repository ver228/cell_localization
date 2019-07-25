#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:58:48 2019

@author: avelinojaver
"""

from pathlib import Path
import pickle

if __name__ == '__main__':
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/woundhealing-v2-mix/different_losses/'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/woundhealing-v2-nuclei/different_losses/'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/eosinophils/20x/eosinophils-20x'
    #root_dir = Path.home() / 'workspace/localization/results/locmax_detection/lymphocytes/20x/lymphocytes-20x'
    root_dir = Path.home() / 'workspace/localization/results/locmax_detection/eggs/worm-eggs-adam'
    
    score_file = root_dir / 'scores.p'
    
    with open(score_file, 'rb') as fid:
        results = pickle.load(fid)
    
    #%%
    
    best_scores = []
    for bn, res in results.items():
        parts = bn.split('_')
        loss_name = parts[2]
        ss = parts[0].rpartition('roi')[-1]
        
        
        
        dd = (ss, loss_name,  res['test_scores']['F1'], res['best_threshold'])
        best_scores.append(dd)
        
    best_scores = sorted(best_scores)
    for dd in best_scores:
        print(dd)
        
        
        