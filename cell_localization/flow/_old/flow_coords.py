#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:27:27 2019

@author: avelinojaver
"""
import tables
import random
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset 
#%%



class CoordsFlow(Dataset):
    def __init__(self, 
                 root_dir,
                 samples_per_epoch = 2000,
                 roi_size = 96,
                 zoom_range = (0.75, 1.5),
                 int_factor_range = (0.5, 1.5),
                 scale_int = (0, 4095),
                 loc_sigma = 1.5
                 ):
        
        
        self.root_dir = Path(root_dir)
        self.samples_per_epoch = samples_per_epoch
        self.roi_size = (roi_size, roi_size)
        
        self.zoom_range = zoom_range
        self.int_factor_range = int_factor_range
        
        self.scale_int = scale_int
        self.loc_sigma = loc_sigma
        
        self.files = [x for x in self.root_dir.rglob('*.hdf5')]
        
        
    def __len__(self):
        return self.samples_per_epoch
            
    
    def __getitem__(self, ind):
        _file = random.choice(self.files)
        img, coords = self.read_data(_file)
        img, coords = self._augment(img, coords)
        img, coords = self._crop(img, coords)
        
        img = (img.astype(np.float32) - self.scale_int[0])/(self.scale_int[1] - self.scale_int[0])
        
        int_factor = random.uniform(*self.int_factor_range)
        img *= int_factor
        img = np.clip(img, 0, 1)
        
        
        coords_mask = coords2mask(coords, img.shape, sigma = self.loc_sigma)

        return img[None], coords_mask[None]
        
    
    def read_data(self, _file):
        with tables.File(str(_file), 'r') as fid:
            img = fid.get_node('/img')[:]
            coords = fid.get_node('/coords')[:].T
        
        
        return img, coords
    
    def _random_sample(self):
        bn = random.choice(self.valid_names)
        file_pairs = random.sample(self.exp_data[bn], 2)
        out = [self._read_image(x) for x in file_pairs]
        return out
    
    
    def _augment(self, X, coord):
        theta = np.random.uniform(-180, 180)
        scaling = 1/np.random.uniform(*self.zoom_range)
        
        X_out, coord_out = affine_transform(X, coord, theta, scaling)
        
        #flips
        if random.random() > 0.5:
            X_out = X_out[::-1]
            coord_out[1] = X_out.shape[0] - coord_out[1]
        
        if random.random() > 0.5:
            X_out = X_out[:, ::-1]
            coord_out[0] = X_out.shape[1] - coord_out[0]
        
        return X_out, coord_out
    
    def _crop(self, img, coords):
        def _is_valid(cc, xlims, ylims):
            val = (cc[0]>= xlims[0]) & (cc[ 0]< xlims[1])
            val &= (cc[1]>= ylims[0]) & (cc[1]< ylims[1])
            return val
            
        _roi_size = np.array(self.roi_size)
        good = _is_valid(coords, (0,img.shape[1]-1), (0,img.shape[0]-1))
        coords = coords[:, good]
        
        r_min = coords.min(axis=1)
        r_max = coords.max(axis=1)
        
        r_min = np.ceil(r_min).astype(np.int)
        r_max = np.floor(r_max).astype(np.int)
        
        #print(coords.shape, r_min.shape)
        
        #crop, c_out = img, coords
        r_min = np.maximum((0,0), r_min - _roi_size)
        r_max = np.minimum(img.shape[::-1] - _roi_size, r_max + _roi_size)
        
        xl = random.randint(r_min[0], r_max[0])
        yl = random.randint(r_min[1], r_max[1])
        
        yr = yl + _roi_size[0]
        xr = xl + _roi_size[1]
        
        
        valid_coords = _is_valid(coords, (xl,xr), (yl,yr))
        
        crop = img[yl:yr, xl:xr]
        offset = np.array((xl, yl))
        c_out = coords[:, valid_coords] - offset[:, None]
        

        
        return crop, c_out
    
    
#%%
def affine_transform(img, coords, theta, scaling = 1., offset_x = 0., offset_y = 0.):
    #It might be faster to resize and then rotate since I am zooming,
    # but I would need to recentre the image and coordinates
    
    cols, rows = img.shape[0], img.shape[1]
    
    M = cv2.getRotationMatrix2D((rows/2,cols/2), theta, scaling)
    
    translation_matrix = np.array([[1, 0, offset_x],
                                   [0, 1, offset_y],
                                   [0, 0, 1]])
    
    M = np.dot(M, translation_matrix)
    dst = cv2.warpAffine(img, M, (rows, cols))#, borderMode = cv2.BORDER_REFLECT_101)
    coords_rot = np.dot(M[:2, :2], coords)  +  M[:, -1][:, None]
    
    return dst, coords_rot   

def coords2mask(coords, roi_size, kernel_size = (25,25), sigma = 4):
    b = np.zeros(roi_size, np.float32)
    
    norm_factor = 2*np.pi*(sigma**2)
    c_cols, c_rows = np.round(coords).astype(np.int)
    c_rows = np.clip(c_rows, 0, roi_size[0]-1)
    c_cols = np.clip(c_cols, 0, roi_size[1]-1)
    b[c_rows, c_cols]  = norm_factor
    
    b = cv2.GaussianBlur(b, kernel_size, sigma, sigma)  
    return b

#%%
if __name__ == '__main__':
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/data/train'
    gen = CoordsFlow(root_dir)
    
    for _ in range(10):
        
        X,Y = gen[0]
        
        plt.figure()
        plt.imshow(X[0], cmap='gray')
        plt.imshow(Y[0], cmap='inferno', alpha=0.5)
        
        
        #fig, axs = plt.subplots(1,2, sharex=True,sharey=True)
        #axs[0].imshow(X, cmap='gray')
        #axs[1].imshow(Y, cmap='gray')
    