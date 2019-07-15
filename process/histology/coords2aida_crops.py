h #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:59:24 2019

@author: avelinojaver
"""

import pandas as pd
from pathlib import Path
import numpy as np
import json
from scipy.spatial.distance import pdist, squareform
#%%
def _remove_redundant(coords, min_dist):
    #%%
    C = coords[['x', 'y']].values
    D = pdist(C)
    
    Dv = squareform(D)
    np.fill_diagonal(Dv, 1e5)
    
    maybe_x, maybe_y = np.where(Dv<5)
    
    pairs = [(x,y) if x < y else (y,x) for x, y in zip(maybe_x, maybe_y)]
    pairs = list(set(pairs))
    
    if not pairs:
        return coords
    
    v_good, v_bad = map(list, zip(*pairs))
    
    coords_r = coords.drop(v_bad)
    #%%
    
    return coords_r

def coords2aida(coords, save_file):
    coords = _remove_redundant(coords, min_dist = 6)
    
    out_dict = {'name' : 'Predictions', 'layers' : []}
    
    names_d = {'L' : 'Lymphocytes Predicted', 'E' : 'Eosinophils Predicted'}
    colors_d = {'L' : {'fill': {'hue': 170,
                   'saturation': 0.44,
                   'lightness': 0.69,
                   'alpha': 0.7},
                  'stroke': {'hue': 170, 'saturation': 0.44, 'lightness': 0.69, 'alpha': 1}},
            'E' : {'fill': {'hue': 60,
               'saturation': 1,
               'lightness': 0.85,
               'alpha': 0.7},
              'stroke': {'hue': 60, 'saturation': 1, 'lightness': 0.85, 'alpha': 1}}
            }
    
     
    
    r_dflt = 8
    hw_dflt = r_dflt*2
    
    for lab, dat in coords.groupby('label'):
        layer_ = {'name' : names_d[lab], 'opacity' : 1, 'items' : []}
        
        
        for _, row in dat.iterrows():
            element_ = {'class': '', 'type': 'circle'}
            element_['color'] = colors_d[lab]
        
            x = row['x']
            y = row['y']
            
            element_['bounds'] = {'x': x,
                                  'y': y,
                                  'width': hw_dflt,
                                  'height': hw_dflt}
            element_['center'] = {'x': x, 'y': y}
            element_['radius'] = r_dflt
            
            layer_['items'].append(element_)
        
        out_dict['layers'].append(layer_)
    
    with open(str(save_file), 'w') as fid:
        dat = json.dump(out_dict, fid)
        

if __name__ == '__main__':
    
    import ast
    import tqdm
    _is_debug = False
    
    bn = 'TH0.1_bladder-tiles-roi64-20x_unetv2_l1smooth_20190529_193223_adam_lr6.4e-05_wd0.0_batch64'
    #coords_dir = Path.home() / 'workspace/localization/predictions/histology_detection' / bn
    coords_dir = Path.home() / 'workspace/localization/predictions/prostate-gland-phenotyping' / bn
    
    coords_files = coords_dir.rglob('*.csv')
    coords_files = [x for x in coords_files if not x.name.startswith('.')]
    coords_files_d = {x.stem:x for x in coords_files}
    
    rois_dir = Path.home() / 'workspace/localization/TILS_candidates_4096/'
    roi_files = rois_dir.glob('*.png')
    
    
    save_dir = rois_dir / 'predictions'
    save_dir.mkdir(parents = True, exist_ok = True)
    
    
    roi_files = [x for x in roi_files if not x.name.startswith('.') ]
    for roi_file in tqdm.tqdm(roi_files):
        
        bn, _, remain = roi_file.stem.partition('_ROI-')
        
        corner, roi_size = remain.split('-')
        corner = ast.literal_eval(corner)
        roi_size = int(roi_size)
        
        if not bn in coords_files_d:
            continue
        
        coordinates_file = coords_files_d[bn]
        
        coords = pd.read_csv(coordinates_file)
        coords.columns = ['label', 'x', 'y']
        
        good = (coords['x'] >= corner[0]) & (coords['x'] <= corner[0] + roi_size)
        good &= (coords['y'] >= corner[1]) & (coords['y'] <= corner[1] + roi_size)
        valid_coords = coords[good].copy()
        if len(valid_coords) == 0:
            continue
        
        valid_coords.reset_index(drop = True, inplace = True)
        valid_coords['x'] -= corner[0]
        valid_coords['y'] -= corner[1]
        
        if _is_debug:
            #%%
            import cv2
            import matplotlib.pylab as plt
            img = cv2.imread(str(roi_file), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            
            plt.figure()
            plt.imshow(img)
            plt.plot(valid_coords['x'], valid_coords['y'], '.r')
            #%%
            break
        
        #%%
        save_file = save_dir / (roi_file.stem + '.json')
        coords2aida(valid_coords, save_file)
        #%%
        
        