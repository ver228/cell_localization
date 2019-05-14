#!/usr/bin/env python3
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
    C = coords[['x', 'y']].values
    D = pdist(C)
    
    Dv = squareform(D)
    np.fill_diagonal(Dv, 1e5)
    
    maybe_x, maybe_y = np.where(Dv<5)
    
    pairs = [(x,y) if x < y else (y,x) for x, y in zip(maybe_x, maybe_y)]
    pairs = list(set(pairs))
    
    v_good, v_bad = map(list, zip(*pairs))
    
    coords_r = coords.drop(v_bad)
    return coords_r

def coords2aida(src_file, save_file):
    coords = pd.read_csv(src_file)
    coords.columns = ['label', 'x', 'y']
    
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
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/predictions'
    root_dir = Path(root_dir)
    
    
    
    for fname in root_dir.glob('*.csv'):
        save_name =  fname.parent / (fname.stem + '.json')
        coords2aida(fname, save_name)
    
        