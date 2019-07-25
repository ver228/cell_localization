#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:40:48 2019

@author: avelinojaver
"""

from openslide import OpenSlide
import cv2

import math
import tables
import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pylab as plt

from torch.utils.data import Dataset, DataLoader

class SlideFlow(Dataset):
    def __init__(self, 
                 fname,
                 roi_size = 1024,
                 roi_pad = 32,
                 slide_level = 0,
                 is_switch_channels = False
                 ):
        print(fname)
        
        self.fname = fname
        self.roi_size = (roi_size, roi_size)
        self.roi_pad = roi_pad
        self.is_switch_channels = is_switch_channels
        
        reader =  OpenSlide(str(self.fname))
        slide_size = reader.dimensions
        
        
        reader.close()
        
        rr = roi_size - roi_pad
        self.corners = [(x,y) for x in range(0, slide_size[0], rr) for y in range(0, slide_size[1], rr)]
        self.slide_size = slide_size
    
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= len(self):
            _out = self[self.n]
            self.n += 1
            return _out
        else:
            raise StopIteration 
    
    def __len__(self):
        return len(self.corners)
            
    
    def __getitem__(self, ind):
        corner = self.corners[ind]
        
        reader =  OpenSlide(str(self.fname))
        roi = reader.read_region(corner, 0, self.roi_size)
        reader.close()
        
        roi = np.array(roi)[..., :-1] #remove the alpha channel
        if self.is_switch_channels:
            roi = roi[..., ::-1]
        
        #roi = np.rollaxis(roi, 2, 0)
        #roi = roi.astype(np.float32)/255
        
        return roi, np.array(corner)
    
if __name__ == '__main__':
    slide_file = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/TMA_counts/2Afirst104834.svs'
    roi_size = 1024
    
    
    
#    binsize = 32
#    n2split = roi_size // binsize
#    reader =  SlideFlow(slide_file, roi_size = roi_size, roi_pad = 0)
#    loader = DataLoader(reader, batch_size = 2, num_workers = 2, collate_fn = lambda x : x)
#    
#    
#    valid_tissue = np.zeros([int(math.ceil(x/roi_size)*n2split) for x in reader.slide_size])
#    
#    for batch in tqdm.tqdm(loader):
#        for img, corner in batch:
#            
#            
#            dd = [np.split(x, n2split, axis=1) for x in np.split(img, n2split, axis = 0)]
#            dd = np.array(dd)
#            dd_std = dd.std(axis=(2,3)).mean(axis=-1)
#            
#            i, j = map(int, corner/binsize)
#            
#            
#            valid_tissue[i:i + dd_std.shape[0], j:j + dd_std.shape[1]] = dd_std
#            
#    
    #%%
    reader =  OpenSlide(str(slide_file))
    
    level = 3
    slide_size = reader.level_dimensions[level]
    img = reader.read_region((0,0), level, slide_size)
    img = np.array(img)[..., :-1]
    #%%
    bw = (img<230).any(axis=2)
    bw = bw.astype(np.uint8)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    
    _, cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [x for x in cnts if cv2.contourArea(x) > 500]
    #%%
    plt.figure()
    plt.imshow(img)
    for cnt in cnts:
        plt.plot(cnt[:, 0, 0], cnt[:, 0, 1])
    plt.show()
    
    #%%
    
#    for ii in tqdm.tqdm(range(0, level_dims[0], roi_size)):
#        rows = []
#        for jj in range(0, level_dims[1], roi_size):
#            corner = (ii, jj)
#            img = reader.read_region(corner, 0, (roi_size, roi_size))
#            img = np.array(img)
#            img = img[..., :-1]
#            
#            
#            dd = [np.split(x, n2split, axis=1) for x in np.split(img, n2split, axis = 0)]
#            dd = np.array(dd)
#            valid_tisue = dd.std(axis=(2,3)).mean(axis=-1)
#            
#            rows.append(valid_tisue)
#        rows = np.concatenate(rows)
#        dat.append(rows)
#    dat =np.concatenate(dat, axis=1)
##    #%%
##    plt.imshow(dat)
#    #%%
    
    
    #img = reader.read_region(corner, 0, (32, 32))
    
    
    
    
    #$$

#    
    