#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:03:09 2019

@author: avelinojaver
"""
from refine_coordinates import correct_coords

import cv2
import pandas as pd
from pathlib import Path
import json
import numpy as np
import tables
import matplotlib.pylab as plt

from matplotlib.widgets import RectangleSelector
#%%

def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    
    print(f"'{fname.name}' : [({round(x1)}, {round(x2)}), ({round(y1)}, {round(y2)})]")
    

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
    if event.key in ['C', 'c']:
        toggle_selector.is_finished = True
    
        
#%%

filters = tables.Filters(complevel=0, 
                          complib='blosc', 
                          shuffle=True, 
                          bitshuffle=True, 
                          fletcher32=True
                          )

if __name__ == '__main__':
    _debug = False
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/'
    root_dir = Path(root_dir)
    
    #annotations_file = root_dir / 'raw'/ 'dot_annotations.csv'
    annotations_file = root_dir / 'raw'/ 'annotations_v2.csv'
    
    save_dir = root_dir / 'data'
    
    (save_dir / 'train').mkdir(exist_ok=True, parents=True)
    (save_dir / 'test').mkdir(exist_ok=True, parents=True)
    
    df = pd.read_csv(str(annotations_file))
    roi_lims = {
                '1.tif' : [(315, 695), (0, 150)],
                '10.tif' : [(0, 553), (0, 511)],
                '11.tif' : [(0, 266), (0, 508)],
                '12.tif' : [(0, 694), (0, 509)],
                '13.tif' : [(25, 690), (0, 508)],
                '14.tif' : [(124, 272), (100, 510)],
                '15.tif' : [(695, 1185), (25, 512),
                            (1185, 1357), (50, 314),
                            (60, 695), (125, 374),
                            (0, 430), (0, 125)],
                '16.tif' : [(75, 363), (243, 508)],
                '17.tif' : [(0, 752), (0, 325),
                            (0, 412), (325, 512)],
                '2.tif' : [(0, 760), (0, 512)],
                '3.tif' : [(0, 566), (0, 512)],
                '4.tif' : [(0, 350), (0, 512)],
                '5.tif' : [(0, 449), (0, 512)],
                '6.tif' : [(0, 695), (0, 512)],
                '7.tif' : [(0, 695), (0, 512)],
                '8.tif' : [(65, 462), (0, 512)],
                '9.tif' : [(0, 267), (0, 321)],
                }
    
    for bn, dat in df.groupby('filename'):
        
        fname = annotations_file.parent / bn
        img = cv2.imread(str(fname), -1)
        
        
        fname_nuclei = fname.parents[1] / 'demixed' / ('N_' + fname.name)
        img_nuclei = cv2.imread(str(fname_nuclei), -1) if fname_nuclei.exists() else None
        
        fname_membrane = fname.parents[1] / 'demixed' / ('M_' + fname.name)
        img_membrane = cv2.imread(str(fname_membrane), -1) if fname_nuclei.exists() else None
        
        #print(img.min(), img.max())
        
        dd = [json.loads(x) for x in dat['region_shape_attributes']]
        coords_ori = np.array([(x['cx'], x['cy']) for x in dd if 'cx' in x])
        
        ind = int(fname.stem)
        if ind in [3, 4, 5]:
            coords = correct_coords(img_nuclei, coords_ori)
        elif ind not in [12, 13, 14, 15, 16, 17]:
            img_b = cv2.blur(img, (5,5))
            coords = correct_coords(img_b, coords_ori)
            
        else:
            coords = coords_ori
            
#        #%%
#        if img_membrane is not None or img_nuclei is not None:
#            fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
#            axs[0].imshow(img, cmap='gray')
#            axs[0].plot(coords[..., 0], coords[..., 1], '.r')
#            
#            if img_nuclei is not None:
#                axs[1].imshow(img_nuclei, cmap='gray')
#                axs[1].plot(coords[..., 0], coords[..., 1], '.r')
#            
#            if img_membrane is not None:
#                axs[2].imshow(img_membrane, cmap='gray')
#                axs[2].plot(coords[..., 0], coords[..., 1], '.r')
#            
#            plt.suptitle(fname.stem)
                  #%%               
        
        test_roi = img.copy()
        valid_test = np.ones(len(coords), np.bool)
        if bn in roi_lims:
        
            for ii in range(0, len(roi_lims[bn]), 2):
            
                xl, yl = roi_lims[bn][ii:ii+2]
                
                train_valid = (coords[:, 0] >= xl[0]) & (coords[:, 0] <= xl[1])
                train_valid &= (coords[:, 1] >= yl[0]) & (coords[:, 1] <= yl[1])
                
                train_coords = coords[train_valid]
                train_coords[..., 0] -= xl[0]
                train_coords[..., 1] -= yl[0]
                
                valid_test &= ~train_valid
                
                train_roi = img[yl[0]:yl[1]+1, xl[0]:xl[1]+1].copy()
                test_roi[yl[0]:yl[1]+1, xl[0]:xl[1]+1] = 0
                
                
                train_coords_df = pd.DataFrame({'type_id':1, 
                                                'cx':train_coords[:, 0], 
                                                'cy':train_coords[:, 1]})
                train_coords_rec = train_coords_df.to_records(index=False)        
        
        
                save_name = save_dir / 'train' / f'{fname.stem}-{ii}.hdf5'
                with tables.File(str(save_name), 'w') as fid:
                    fid.create_carray('/', 'img', obj = train_roi, filters  = filters)
                    fid.create_table('/', 'coords', obj = train_coords_rec)
                
                
                if img_membrane is not None:
                    save_name = save_dir / 'train' / f'M_{fname.stem}-{ii}.hdf5'
                    with tables.File(str(save_name), 'w') as fid:
                        _roi = img_membrane[yl[0]:yl[1]+1, xl[0]:xl[1]+1].copy()
                        fid.create_carray('/', 'img', obj = _roi)
                        fid.create_table('/', 'coords', obj = train_coords_rec, filters  = filters)
                    
                if img_nuclei is not None:
                    save_name = save_dir / 'train' / f'N_{fname.stem}-{ii}.hdf5'
                    with tables.File(str(save_name), 'w') as fid:
                        _roi = img_nuclei[yl[0]:yl[1]+1, xl[0]:xl[1]+1].copy()
                        fid.create_carray('/', 'img', obj = _roi, filters  = filters)
                        fid.create_table('/', 'coords', obj = train_coords_rec)
                
                
                test_coords = coords[valid_test]
                if _debug:
                    fig, axs = plt.subplots(1, 2)
                    axs[0].imshow(train_roi, cmap='gray')
                    axs[0].plot(train_coords[..., 0], train_coords[..., 1], '.r')
                    
                    #axs[1].imshow(np.log(train_roi+1), cmap='gray')
                    axs[1].imshow(test_roi, cmap='gray')
                    axs[1].plot(test_coords[..., 0], test_coords[..., 1], '.b')
                    plt.suptitle(bn)
            
            save_name = save_dir / 'test' / (fname.stem + '.hdf5')
            with tables.File(str(save_name), 'w') as fid:
                fid.create_carray('/', 
                                  'img', 
                                  obj = test_roi, 
                                  filters  = filters
                                  )
                
                if test_coords.size > 0:
                    test_coords_df = pd.DataFrame({'type_id':1, 
                                                'cx':test_coords[:, 0], 
                                                'cy':test_coords[:, 1]})
                    test_coords_rec = test_coords_df.to_records(index=False) 
                    fid.create_table('/', 'coords', obj = test_coords_rec)
        
        
        else:
            continue
            if _debug:
                fig, ax = plt.subplots(1, 1)
                ax.imshow(img, cmap='gray')
                ax.plot(coords[..., 0], coords[..., 1], '.r')
                
                
                toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)
                toggle_selector.is_finished = False
                
                plt.connect('key_press_event', toggle_selector)
                plt.show()
                
                while not toggle_selector.is_finished:
                    plt.pause(0.5)
                
                plt.close()