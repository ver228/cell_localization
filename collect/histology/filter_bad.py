#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:21:30 2019

@author: avelinojaver
"""
import pandas as pd
from pathlib import Path
#%%
if __name__ == '__main__':
    src_file = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/Key to images sent to Avelino June 2019.xlsx'
    df = pd.read_excel(src_file)
    
    rois_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/lymphocytes/40x'
    rois_dir_bad = str(rois_dir).replace('/lymphocytes/', '/bad_lymphocytes/')
    
    roi_files = list(rois_dir.rglob('*.hdf5'))
    
    mirax_folder = Path.home() / 'projects/bladder_cancer_tils/raw/HEs/'
    mirax_bn = [x.stem for x in mirax_folder.rglob('*.mrxs')]
    
    valid_bn = []
    bad_bn = mirax_bn
    for _, row in df.iterrows():
        bn = row['Slide ID'][:-4]
        if row['Use?'] == 'Yes':
            valid_bn.append(bn)
        else:
            bad_bn.append(bn)
    
    
    missing_bn = []
    
    bad_files = []
    for fname in roi_files:
        parts = fname.stem.split('_')
        if parts[-1] == 'resized':
            parts = parts[:-1]
                
        if parts[0] in ['E', 'L']:
            dd = parts[1:-1]
        else:
            dd = parts[:-3]
        bn = '_'.join(dd)
        
        if bn in bad_bn:
            bad_files.append(fname)
        
        elif not bn in valid_bn:
        
            missing_bn.append((bn, fname.stem))
        
    assert not missing_bn
    #%%
    for fname in bad_files:
        new_fname = Path(str(fname).replace(str(rois_dir), rois_dir_bad))
        
        new_fname.parent.mkdir(parents = True, exist_ok = True)
        fname.rename(new_fname)
    
    