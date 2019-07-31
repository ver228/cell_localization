#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 20:31:55 2019

@author: avelinojaver
"""
from pathlib import Path
import tables
import tqdm

if __name__ == '__main__':
    root_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils'
    empty_dir = Path.home() / 'workspace/localization/data/histology_bladder/empty'
    empty_dir.mkdir(parents = True, exist_ok = True)
    
    bad_files = []
    
    for fname in tqdm.tqdm(root_dir.rglob('*.hdf5')):
        with tables.File(fname, 'r') as fid:
            if not '/coords' in fid:
                fname_new = empty_dir / fname.name
                fname.rename(fname_new)
            
    
    