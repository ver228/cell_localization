#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:03:09 2019

@author: avelinojaver
"""

from pathlib import Path
import json
import numpy as np
import random
import tqdm
from openslide import OpenSlide
import cv2

#%%

if __name__ == '__main__':
    _debug = False
    roi_sizes_d = {'20' : 512, '40' : 1024}
    
    root_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/')
    
    slides_dir = root_dir / 'raw'
    annotations_dir = root_dir / 'eosinophils_regions'
    save_dir = root_dir / 'eosinophils_sampled_rois'
    
    
    slides_files_d = {x.stem : x for x in slides_dir.rglob('*.svs')}
    
    annotation_files = list(annotations_dir.rglob('*.json'))
    for annotation_file in tqdm.tqdm(annotation_files):
        bn = annotation_file.stem.partition(' ')[0]
        slide_file = slides_files_d[bn]
    
        with open(annotation_file) as fid:
            annotations = json.load(fid)
        
        
        
        reader =  OpenSlide(str(slide_file))
        objective = reader.properties['openslide.objective-power']
        vendor = reader.properties['openslide.vendor']
        roi_size = roi_sizes_d[objective]
        roi_half = roi_size // 2
        
        cnts = [np.array(cnt, dtype=np.float32) for name, cnt in annotations if name != 'Whitespace']
        
        level_n = min(reader.level_count - 1, 7)
        level_dims = reader.level_dimensions[level_n]
        downsample = reader.level_downsamples[level_n]
        corner = (0,0)
        img_small = reader.read_region(corner, level_n, level_dims)
        
        for icnt, cnt in enumerate(cnts):
            #area = cv2.contourArea(cnt)
            bbox = cv2.boundingRect(cnt)
            
            xl, xr = bbox[0], bbox[0] + bbox[2]
            yl, yr = bbox[1], bbox[1] + bbox[3]
            
            n_samples = 3
            prev_dist = []
            for isample in range(n_samples):
                
                cnt_lim = min(bbox[2], bbox[3])//4
                cnt_lim = min(roi_half//2, cnt_lim)
                
                n_test = 100
                dist2cnt = -1
                
                if prev_dist:
                    prev_x, prev_y = map(np.array, zip(*prev_dist))
                
                for i_test in range(200):
                    xc = random.uniform(xl, xr)
                    yc = random.uniform(yl, yr)
                    
                    if prev_dist:
                        _dist = np.sqrt((prev_x-xc)**2 + (prev_y - yc)**2)
                        _dist = np.min(_dist)
                        if _dist < roi_size:
                            continue
                    
                    dist2cnt = cv2.pointPolygonTest(cnt, (xc, yc), measureDist = True)
                    if dist2cnt >= cnt_lim:
                        break
                else:
                    break #I have problems to find a good sample do not bother anymore
                    
                
                prev_dist.append((xc, yc))
                
                xl = int(xc - roi_half)
                yl = int(yc - roi_half)
                corner = (xl, yl)
                
                img_o = reader.read_region(corner, 0, (roi_size, roi_size)) 
                
                save_name = save_dir / f'S{isample}_{bn}_{vendor}-{objective}_CNT{icnt}_C{corner[0]}-{corner[1]}_R{roi_size}.png'
                
                save_name.parent.mkdir(exist_ok = True, parents = True)
                img_o = np.array(img_o)[..., -2::-1]
                
                cv2.imwrite(str(save_name), img_o)
                
                if _debug:
                    import matplotlib.pylab as plt
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(img_small)
                    cnt_small = cnt/downsample
                    bb_small = np.array(bbox)/downsample
                    axs[0].plot(cnt_small[:, 0], cnt_small[:, 1], 'g')
                    axs[0].plot(xc/downsample, yc/downsample, 'or')
                    axs[0].set_xlim(bb_small[0] - 20, bb_small[0] + bb_small[2] + 20 )
                    axs[0].set_ylim(bb_small[1] - 20, bb_small[1] + bb_small[3] + 20 )
                    
                    axs[1].imshow(img_o[..., ::-1])
                
                
            
            
    #%%
    
    