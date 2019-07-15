#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:50:47 2019

@author: avelinojaver
"""

from pathlib import Path
import tqdm
import shutil

if __name__ == '__main__':
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/'
    root_dir = Path(root_dir)
    
    src_dir = root_dir / 'raw'
    assert root_dir.exists()
    
    filtered_rois_dir = root_dir / 'filtered_crops' / 'x200-y1-roi256'
    assert root_dir.exists()
    
    
    dst_dir = root_dir / 'manually_filtered'
    
    #%%
    #filter files. I took two crops from each of the original files. I then deleted the 
    #crops that didn't seem to match the folder (nuclei, membrane or nuclei_and_membrane)
    #I will only keep a file in a given folder if both of the crops remained...
    fnames = filtered_rois_dir.rglob('ROI-LOG-POS*.png')
    pos_basenames = []
    for fname in fnames:
        dd = (fname.parent.name, fname.stem.partition('_')[-1])
        pos_basenames.append(dd)
    
    fnames = filtered_rois_dir.rglob('ROI-LOG-NEG*.png')
    neg_basenames = []
    for fname in fnames:
        dd = (fname.parent.name, fname.stem.partition('_')[-1])
        neg_basenames.append(dd)
    basenames = set(pos_basenames) & set(neg_basenames)
    
    #%%
    src_fnames = fnames = src_dir.rglob('*.tif')
    src_dict = {x.stem : x for x in src_fnames}
    
    for dclass, bn in basenames:
        fname = src_dict[bn]
        loc_dst_dir = dst_dir / dclass
        
        loc_dst_dir.mkdir(exist_ok = True, parents = True)
        shutil.copy(fname, loc_dst_dir)
        
        
    
    
    
    
    
    
    
    