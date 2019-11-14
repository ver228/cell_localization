#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:27:27 2019

@author: avelinojaver
"""
#import multiprocessing as mp
#mp.set_start_method('spawn', force=True)

import math
import tables
import random
from pathlib import Path
import numpy as np

import tqdm

from torch.utils.data import Dataset 

from .transforms import RandomCropWithSeeds, AffineTransform, RemovePadding, RandomVerticalFlip, \
RandomHorizontalFlip, NormalizeIntensity, RandomIntensityOffset, RandomIntensityExpansion, \
FixDTypes, ToTensor, Compose
           
class RandomFlatField(object):
    def __init__(self, roi_size):
        self.roi_size = roi_size
        
        self.sigma_range = roi_size//2, roi_size*4
        self.xx, self.yy = np.meshgrid(np.arange(roi_size), np.arange(roi_size))
        
    def __call__(self, image, target):
        
        base_factor = random.uniform(0.0, 0.5)
        aug_factor = random.uniform(0.8, 2)
        sigma = random.uniform(*self.sigma_range )
        
        mu_x = random.uniform(0, self.roi_size)
        mu_y = random.uniform(0, self.roi_size)
        dx = self.xx-mu_x
        dy = self.yy-mu_y
        
        flat_field = np.exp(-(dx*dx + dy*dy)/sigma**2)
        flat_field -= flat_field.min()
        flat_field = aug_factor*(1 - base_factor) * flat_field + base_factor
        image = image*flat_field
        
        return image, target
     

class MasksFlow(Dataset):
    def __init__(self, 
                 data_src,
                 folds2include = None,
                 num_folds = 5,
                 samples_per_epoch = 2000,
                 roi_size = 96,
                 scale_int = (0, 255),
                 norm_mean = 0.,
                 norm_sd = 1.,
                 zoom_range = (0.90, 1.1),
                 int_aug_offset = None,
                 int_aug_expansion = None,
                 
                 max_input_size = 2048 #if any image is larger than this it will be splitted (only working with folds)
                 ):
        
        _dum = set(dir(self))
        
        
        self.data_src = Path(data_src)
        if not self.data_src.exists():
            raise ValueError(f'`data_src` : `{data_src}` does not exists.')
        
        
        self.folds2include  = folds2include
        self.num_folds = num_folds
        
        self.samples_per_epoch = samples_per_epoch
        self.roi_size = roi_size
        self.scale_int = scale_int
        
        self.norm_mean = norm_mean
        self.norm_sd = norm_sd
        
        self.zoom_range = zoom_range
        self.int_aug_offset = int_aug_offset
        self.int_aug_expansion = int_aug_expansion
        self.max_input_size = max_input_size
        
        
        rotation_pad_size = math.ceil(self.roi_size*(math.sqrt(2)-1)/2)
        padded_roi_size = roi_size + 2*rotation_pad_size
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        prob_unseeded_patch = 0.
        
        transforms_random = [
                RandomCropWithSeeds(padded_roi_size, rotation_pad_size, prob_unseeded_patch),
                AffineTransform(zoom_range),
                RemovePadding(rotation_pad_size),
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                NormalizeIntensity(scale_int, norm_mean, norm_sd),
                RandomFlatField(roi_size),
                RandomIntensityOffset(int_aug_offset),
                RandomIntensityExpansion(int_aug_expansion),
                FixDTypes(),
                ToTensor()
                #I cannot really pass the ToTensor to the dataloader since it crashes when the batchsize is large (>256)
                ]
        self.transforms_random = Compose(transforms_random)
        
        transforms_full = [
                NormalizeIntensity(scale_int),
                FixDTypes(),
                ToTensor()
                ]
        self.transforms_full = Compose(transforms_full)
        
        self.data = self.load_data_from_file(self.data_src)
         
        
        self.type_ids = sorted(list(self.data.keys()))
        self.types2label = {k:(ii + 1) for ii, k in enumerate(self.type_ids)}
        
        self.num_clases = len(self.type_ids)
        
        #flatten data so i can access the whole list by index
        self.data_indexes = [(_type, _fname, ii)for _type, type_data in self.data.items() 
                                    for _fname, file_data in type_data.items() for ii in range(len(file_data))]
        
        assert len(self.data_indexes) > 0 #makes sure there are valid files
    
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
    
    def load_data_from_file(self, src_file):
        
        def is_in_fold(_id, fold):
            return ( (_id - 1) % self.num_folds ) == (fold - 1)
        
        if self.folds2include is None:
            _is_valid_fold = lambda x : True
        elif isinstance(self.folds2include, (int, float)):
            assert self.folds2include >= 1 and self.folds2include <= self.num_folds
            _is_valid_fold = lambda x : is_in_fold(x, self.folds2include)
        else:
            assert all([x >= 1 and x <= self.num_folds for x in self.folds2include])
            _is_valid_fold = lambda x : any([is_in_fold(x, fold) for fold in self.folds2include])
        
        #for the moment i am not sure how to deal with the type, so i am assuming a single class
        
        tid = 1
        data = {tid : {}}
        with tables.File(str(src_file), 'r') as fid:
            
            src_files = fid.get_node('/src_files')[:]
            
            images = fid.get_node('/images')
            masks = fid.get_node('/masks')
            
            for row in src_files:
                file_id = row['file_id']
                if not _is_valid_fold(file_id):
                    continue
                
                ii = file_id - 1
                img = images[ii]
                mask = masks[ii]
                
                if np.all([x <= self.max_input_size for x in img.shape[:-1]]):
                
                    data[tid][file_id] = [(img, mask)]
                else:
                    
                    inds = []
                    for ss in img.shape[:-1]:
                        n_split = int(np.ceil(ss/self.max_input_size)) + 1
                        ind = np.linspace(0, ss, n_split)[1:-1].round().astype(np.int)
                        inds.append([0, *ind.tolist(), ss])
                    
                    inds_i, inds_j = inds
                    for i1, i2 in zip(inds_i[:-1], inds_i[1:]):
                        for j1, j2 in zip(inds_j[:-1], inds_j[1:]):
                            roi = img[i1:i2, j1:j2]
                            roi_mask = mask[i1:i2, j1:j2]
                            
                            data[tid][file_id].append((roi, roi_mask))
                
        return data
                

    def __len__(self):
        return self.samples_per_epoch
            
    
    def __getitem__(self, ind):
        return self.read_random()
        
        

    def read_full(self, ind):
        (_type, _group, _img_id) = self.data_indexes[ind]
        img, target = self.read_key(_type, _group, _img_id)
        img, target = self.transforms_full(img, target)
        
        return img, target['mask']
    
    def read_key(self, _type, _group, _img_id):
        img, mask = self.data[_type][_group][_img_id]
        
        target = dict(mask = mask)
        
        return img, target
    
        
    
    def read_random(self):
        _type = random.choice(self.type_ids)
        _group = random.choice(list(self.data[_type].keys()))
        _img_id = random.randint(0,len(self.data[_type][_group])-1)
        
        img, target = self.read_key(_type, _group, _img_id)
        
        img, target = self.transforms_random(img, target)

        return img,  target['mask']


    
#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt

    src_file = '/Volumes/rescomp1/data/segmentation/data/wound_area_masks.hdf5'
    
    
    
    flow_args = dict(
                roi_size = 256,
                scale_int = (0, 4095.),
                
                zoom_range = (0.5, 1.1),
                
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3),
                
                folds2include = 1#[1,2,3,4]
                )


    gen = MasksFlow(src_file, **flow_args)
    
#%%
    col_dict = {1 : 'r', 2 : 'g'}
    for _ in tqdm.tqdm(range(10)):
        X, mask = gen[0]
        
        #X = X.numpy()
        if X.shape[0] == 3:
            #x = X[::-1]
            x = X
            x = np.rollaxis(x, 0, 3)
        else:
            mask = mask[0]
            x = X[0]
        
        
        
#        xd = x.numpy()
#        
#        fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
#        axs[0].imshow(r)
#        axs[1].imshow(xd*r)
        
        fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
        axs[0].imshow(x,  cmap='gray')
        axs[1].imshow(mask,  cmap='gray', vmin = 0.0, vmax = 1.0)
        #%%
#        for lab in np.unique(labels):
#            good = labels == lab
#            plt.plot(coords[good, 0], coords[good, 1], 'o', color = col_dict[lab])
        
