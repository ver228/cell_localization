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



class MergeFlow(Dataset):
    def __init__(self, 
                 ch1_dir,
                 ch2_dir,
                 samples_per_epoch = 2000,
                 roi_size = 96,
                 zoom_range = (0.75, 1.5),
                 int_factor_range = (0.2, 1.2),
                 int_base_range = (0., 0.2),
                 scale_int = (0, 4095),
                 loc_sigma = 1.5,
                 mix_factor = 0.5
                 ):
        
        self.ch1_dir = Path(ch1_dir)
        self.ch2_dir = Path(ch2_dir)
        
        self.samples_per_epoch = samples_per_epoch
        self.roi_size = (roi_size, roi_size)
        
        self.zoom_range = zoom_range
        self.int_factor_range = int_factor_range
        self.int_base_range = int_base_range
        
        self.scale_int = scale_int
        self.loc_sigma = loc_sigma
        
        self.mix_factor = mix_factor
        
        #print(self.ch1_dir)
        #print(self.ch2_dir)

        self.ch1_files = [x for x in self.ch1_dir.rglob('*.tif') if not x.name.startswith('.')]
        self.ch2_files = [x for x in self.ch2_dir.rglob('*.tif') if not x.name.startswith('.')]
        
        
        
    def __len__(self):
        return self.samples_per_epoch
            
    
    def __getitem__(self, ind):
        ch1_img = self._read_sample(self.ch1_files)
        ch2_img = self._read_sample(self.ch2_files)
        
        
        
        Xin = self.mix_factor*ch1_img + (1 - self.mix_factor)*ch2_img
        
        int_base = random.uniform(*self.int_base_range)
        Xin += int_base
        
        Xout = np.stack((ch1_img, ch2_img))
        
        
        return Xin[None], Xout
    
    def _read_sample(self, _files):
        #%%
        _file = random.choice(_files)
        img = cv2.imread(str(_file), -1)
        img = img.astype(np.float32)
        
        img = self._augment(img)
        img = self._crop(img)
        
        
        #bot, top = img.min(), img.max()
        bot, top = np.percentile(img, [1, 99])
        img = (img.astype(np.float32) - bot)/(top - bot)
        
        
        int_factor = random.uniform(*self.int_factor_range)
        img *= int_factor
        
        
        return img
    
    
    
    def _augment(self, X):
        
        theta = np.random.uniform(-180, 180)
        scaling = 1/np.random.uniform(*self.zoom_range)
        
        cols, rows = X.shape[0], X.shape[1]
        M = cv2.getRotationMatrix2D((rows/2,cols/2), theta, scaling)
        
        translation_matrix = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
        
        M = np.dot(M, translation_matrix)
        X_out = cv2.warpAffine(X, M, (rows, cols), borderMode = cv2.BORDER_REFLECT_101)
            
        #flips
        if random.random() > 0.5:
            X_out = X_out[::-1]
        
        if random.random() > 0.5:
            X_out = X_out[:, ::-1]
          
        #%%
        return X_out
    
    def _crop(self, img):
        xl = random.randint(0, img.shape[1] - self.roi_size[1])
        yl = random.randint(0, img.shape[0] - self.roi_size[0])
        
        yr = yl + self.roi_size[0]
        xr = xl + self.roi_size[1]
        
        crop = img[yl:yr, xl:xr]
        
        return crop
    

#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/cell_demixer/'
    root_dir = Path(root_dir)
    
    ch1_dir = root_dir / 'nuclei'
    ch2_dir = root_dir / 'membrane'
    
    gen = MergeFlow(ch1_dir, ch2_dir)
    
    for ii, (X,Y) in enumerate(gen):
        
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        axs[0].imshow(X[0])
        axs[1].imshow(Y[0])
        axs[2].imshow(Y[1])
        
        if ii > 10:
            break
    
    
#    fnames = root_dir.rglob('*.tif')
#    
#    #merged = [1, 2, 3, 4, 5, 6, 7, 8, 9]
#    #membrane = [12, 13, 14, 15, 16, 17]
#    #nuclei = [10, 11]
#    
#    for fname in fnames:
#        img = cv2.imread(str(fname), -1)
#        
#        plt.figure()
#        plt.imshow(img, cmap='gray')
#        plt.title(fname.name)
#        
        
        
        #fig, axs = plt.subplots(1,2, sharex=True,sharey=True)
        #axs[0].imshow(X, cmap='gray')
        #axs[1].imshow(Y, cmap='gray')
    