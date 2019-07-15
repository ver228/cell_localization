#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:02:23 2019

@author: avelinojaver
"""
import os
import shutil
from pathlib import Path
import tqdm

root_dir = Path.home() / 'workspace/localization/TILS_candidates_4096/deepzoom/'

grouped_data = {}
for fname in root_dir.glob('*.dzi'):
    bn = fname.stem
    if bn.startswith('.'):
        pass
    
    bn = bn.split('_ROI')[0]
    
    if not bn in grouped_data:
        grouped_data[bn] = []
        
    grouped_data[bn].append(fname)
    
#%%
for k, fnames in tqdm.tqdm(grouped_data.items()):
    if len(fnames) > 1:
        for fname in fnames[1:]:
            dname = fname.parent / (fname.stem + '_files')
            
            shutil.rmtree(dname)
            os.remove(fname)