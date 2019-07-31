#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 13:14:48 2019

@author: avelinojaver
"""

import pandas as pd
from pathlib import Path
import cv2
from matplotlib.pylab import plt
import numpy as np

if __name__ == '__main__':
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/raw_separate_channels/'
    root_dir = Path(root_dir)
    
    
    GT_noisy = pd.read_excel(root_dir / 'groundTruthForAvelino_30.xlsx')
    img_ids = pd.read_excel(root_dir / 'imageIDsForAvelino_30.xlsx')
    #%%
    for _, row in img_ids.iterrows():
        img_id = row['ImageNumber']
        coords = GT_noisy[GT_noisy['ImageNumber'] == img_id]
        
        img_path = root_dir / 'nuclei' / row['Nuclear']
        if not img_path.exists():
            continue
        
        img = cv2.imread(str(img_path), -1)
        
        xx, yy = coords['Location_Center_X'], coords['Location_Center_Y']
        
        plt.figure()
        plt.imshow(np.log(img+1), 'gray')
        plt.plot(xx, yy, '.r')
        
        
    
    