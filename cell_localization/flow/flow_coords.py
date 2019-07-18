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
import cv2

import tqdm

import torch
from torch.utils.data import Dataset 

def collate_pandas(batch):
    X, coordinates, labels = zip(*batch)
    X = torch.from_numpy(np.stack(X))
    
    targets = []
    for labs, coords in zip(coordinates, labels):
        t = {
                'coordinates' : torch.from_numpy(coords).long(),
                'labels' : torch.from_numpy(labs).long()
                }
        targets.append(t)
        
    
    X = torch.from_numpy(np.stack(X)).float()
    
    return X, targets


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomCropWithSeeds(object):
    def __init__(self, padded_roi_size = 512, padding_size = 5, prob_unseeded = 0.5):
        self.prob_unseeded = prob_unseeded
        self.padded_roi_size = padded_roi_size
        self.padding_size = padding_size

    def get_seed(self, img_dims, target):
        seed = None
        if random.random() > self.prob_unseeded:
            _type = random.choice(np.unique(target['labels']))
            #randomly select a type
            coords = target['coordinates'][target['labels'] == _type]
            
            yl, xl = img_dims
            x, y = coords.T
            
            good = (x > self.padding_size) & (x  < (xl - self.padding_size)) #[ppp-----rrrrrppp] where p is the padding and r + p is the roi_padded roi_size
            good &= (y > self.padding_size) & (y < (yl - self.padding_size))
            coords = coords[good]
            
            if len(coords) > 0:
                seed = [int(round(x)) for x in random.choice(coords)]
            
        
        return seed
    def __call__(self, image, target):
        
        #### select the limits allowed for a random crop
        xlims = (0, image.shape[1] - self.padded_roi_size - 1)
        ylims = (0, image.shape[0] - self.padded_roi_size - 1)
        
        seed = self.get_seed(image.shape[:2], target)
        if seed is not None:
            p = self.padded_roi_size
            x,y = seed
            
            seed_xlims = (x - p, x)
            seed_xlims = list(map(int, seed_xlims))
            
            seed_ylims = (y - p, y)
            seed_ylims = list(map(int, seed_ylims))
            
            x1 = max(xlims[0], seed_xlims[0])
            x2 = min(xlims[1], seed_xlims[1])
            
            y1 = max(ylims[0], seed_ylims[0])
            y2 = min(ylims[1], seed_ylims[1])
            
            
            xlims = x1, x2
            ylims = y1, y2
            
            
        #### crop with padding in order to keep a valid rotation 
        xl = random.randint(*xlims)
        yl = random.randint(*ylims)
        
        yr = yl + self.padded_roi_size
        xr = xl + self.padded_roi_size
    
        image_crop = image[yl:yr, xl:xr].copy()
        
        
        xx, yy = target['coordinates'].T
        valid = (xx > xl) & (xx < xr)
        valid &= (yy > yl) & (yy < yr)
        
        #I will include labels <= 0 to indicate coordinate from hard negative mining
        valid &= target['labels'] > 0
        
        coordinates = target['coordinates'][valid].copy()
        coordinates[:, 0] -= xl
        coordinates[:, 1] -= yl
        
        
        target['coordinates'] = coordinates
        target['labels'] = target['labels'][valid].copy()
        
        return image_crop, target


class AffineTransform():
    def __init__(self, zoom_range = (0.9, 1.1), rotation_range = (-90, 90)):
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
    
    def __call__(self, image, target):
        theta = np.random.uniform(*self.rotation_range)
        scaling = 1/np.random.uniform(*self.zoom_range)
        
        
        cols, rows = image.shape[0], image.shape[1]
    
        M = cv2.getRotationMatrix2D((rows/2,cols/2), theta, scaling)
        
        offset_x = 0.
        offset_y = 0.
        translation_matrix = np.array([[1, 0, offset_x],
                                       [0, 1, offset_y],
                                       [0, 0, 1]])
        
        M = np.dot(M, translation_matrix)
        
        if image.ndim == 2:
            image_rot = cv2.warpAffine(image, M, (rows, cols))
        else:
            image_rot = [cv2.warpAffine(image[..., n], M, (rows, cols)) for n in range(image.shape[-1])] 
            image_rot = np.stack(image_rot, axis=2)
            
        coords = target['coordinates'].T
        coords_rot = np.dot(M[:2, :2], coords)  +  M[:, -1][:, None]
        coords_rot = np.round(coords_rot).T
        
        target['coordinates'] = coords_rot
        
        
        return image_rot, target

class RemovePadding():
    def __init__(self, pad_size):
        self.pad_size = pad_size
    
    def __call__(self, image, target):
        ##### remove padding
        img_out = image[self.pad_size:-self.pad_size, self.pad_size:-self.pad_size].copy()
        coord_out = target['coordinates'] - self.pad_size
        
        valid = (coord_out[:, 0] >= 0) & (coord_out[:, 0] < img_out.shape[1])
        valid &= (coord_out[:, 1] >= 0) & (coord_out[:, 1] < img_out.shape[0])
        coord_out = coord_out[valid].copy()
        
        target['coordinates'] = coord_out
        target['labels'] = target['labels'][valid].copy()
        
        return img_out, target
    
class RandomVerticalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[::-1]
            
            coordinates = target['coordinates'].copy()
            coordinates[:, 1] = (image.shape[0] - 1 - coordinates[:, 1])
            target['coordinates'] = coordinates
        
        return image, target
    
class RandomHorizontalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[:, ::-1]
            
            coordinates = target['coordinates'].copy()
            coordinates[:, 0] = (image.shape[1] - 1 - coordinates[:, 0])
            target['coordinates'] = coordinates
        
        return image, target

class NormalizeIntensity(object):
    def __init__(self, scale = (0, 255)):
        self.scale = scale
    
    def __call__(self, image, target):
        
        image = (image.astype(np.float32) - self.scale[0])/(self.scale[1] - self.scale[0])
        
        return image, target
        

class RandomIntensityOffset(object):
    def __init__(self, offset_range = (-0.2, 0.2)):
        self.offset_range = offset_range
    
    def __call__(self, image, target):
        if self.offset_range is not None and random.random() > 0.5:
            offset = random.uniform(*self.offset_range)
            image = image + offset
            
        return image, target


class RandomIntensityExpansion(object):
    def __init__(self, expansion_range = (0.7, 1.3)):
        self.expansion_range = expansion_range
    
    def __call__(self, image, target):
        if self.expansion_range is not None and random.random() > 0.5:
            factor = random.uniform(*self.expansion_range)
            image = image*factor
            
        return image, target    

class FixDTypes(object):
    def __call__(self, image, target):
        if image.ndim == 3:
            image = np.rollaxis(image, 2, 0).copy()
        else:
            image = image[None]
        
        image = image.astype(np.float32)
        target['coordinates'] = target['coordinates'].astype(np.int)
        target['labels'] = target['labels'].astype(np.int)
        
        return image, target
    
class ToTensor(object):
    def __call__(self, image, target):
        
        image = torch.from_numpy(image).float()
        target['coordinates'] = torch.from_numpy(target['coordinates']).long()
        target['labels'] = torch.from_numpy(target['labels']).long()
        
        return image, target
#%%
class CoordFlow(Dataset):
    def __init__(self, 
                 root_dir,
                 samples_per_epoch = 2000,
                 roi_size = 96,
                 scale_int = (0, 255),
                 zoom_range = (0.90, 1.1),
                 prob_unseeded_patch = 0.2,
                 int_aug_offset = None,
                 int_aug_expansion = None,
                 is_preloaded = True
                 ):
        
        
        
        
        self.root_dir = Path(root_dir)
        self.samples_per_epoch = samples_per_epoch
        
        self.roi_size = roi_size
        rotation_pad_size = math.ceil(self.roi_size*(math.sqrt(2)-1)/2)
        padded_roi_size = roi_size + 2*rotation_pad_size
        
        self.zoom_range = zoom_range
        self.scale_int = scale_int
        self.prob_unseeded_patch = prob_unseeded_patch
        
        self.int_aug_offset = int_aug_offset
        self.int_aug_expansion = int_aug_expansion
        
        transforms_random = [
                RandomCropWithSeeds(padded_roi_size, rotation_pad_size, prob_unseeded_patch),
                AffineTransform(zoom_range),
                RemovePadding(rotation_pad_size),
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                NormalizeIntensity(scale_int),
                RandomIntensityOffset(int_aug_offset),
                RandomIntensityExpansion(int_aug_expansion),
                FixDTypes()
                ]
        self.transforms_random = Compose(transforms_random)
        
        transforms_full = [
                NormalizeIntensity(scale_int),
                FixDTypes(),
                ToTensor()
                ]
        self.transforms_full = Compose(transforms_full)
        
        
        self.hard_neg_data = None
        self.is_preloaded = is_preloaded
        self.data = self.load_data(self.root_dir, padded_roi_size, self.is_preloaded)
        self.type_ids = sorted(list(self.data.keys()))
        self.num_clases = len(self.type_ids)
        
        
        #flatten data so i can access the whole list by index
        self.data_indexes = [(_type, _fname, ii)for _type, type_data in self.data.items() 
                                    for _fname, file_data in type_data.items() for ii in range(len(file_data))]
        
        assert len(self.data_indexes) > 0 #makes sure there are valid files
        
    
    def load_data(self, root_dir, padded_roi_size, is_preloaded = True):
        data = {} 
        fnames = [x for x in root_dir.rglob('*.hdf5') if not x.name.startswith('.')]
        for fname in tqdm.tqdm(fnames, 'Preloading Data'):
            with tables.File(str(fname), 'r') as fid:
                img = fid.get_node('/img')
                img_shape = img.shape[:2]
                
                if any([x < padded_roi_size for x in img_shape]):
                    continue
                
                if not '/coords' in fid:
                    continue
                rec = fid.get_node('/coords')[:]
                
                if is_preloaded:
                    x2add = (img[:], rec)
                else:
                    x2add = fname
            
            type_ids = set(np.unique(rec['type_id']).tolist())
            k = fname.parent.name
            for tid in type_ids:
                if tid not in data:
                    data[tid] = {}
                if k not in data[tid]:
                    data[tid][k] = []
                
                data[tid][k].append(x2add)
        
        return data

    def __len__(self):
        return self.samples_per_epoch
            
    
    def __getitem__(self, ind):
        #seems like python native multiprocessing does a complete copy to a list or dict https://github.com/pytorch/pytorch/issues/13246
        #this problem can caused crashes due to lack in the share memory. I will try converting the data into a pandas
        img, target = self.read_random()
        
        labels = target['labels'].copy()
        coordinates = target['coordinates'].copy()
        del target
        
        return img, labels, coordinates

    def read_full(self, ind):
        (_type, _group, _img_id) = self.data_indexes[ind]
        img, target = self.read_key(_type, _group, _img_id)
        img, target = self.transforms_full(img, target)
        
        return img, target
    
    def read_key(self, _type, _group, _img_id):
        input_ = self.data[_type][_group][_img_id]
        
        if self.is_preloaded:
           img, coords_rec = input_
        else:
            fname = input_
            with tables.File(str(fname), 'r') as fid:
                coords_rec = fid.get_node('/coords')[:]
                img = fid.get_node('/img')[:]
        
        target = dict(
                labels = coords_rec['type_id'],
                coordinates = np.array((coords_rec['cx'], coords_rec['cy'])).T
                )
        return img, target
        
    
    def read_random(self):
        _type = random.choice(self.type_ids)
        _group = random.choice(list(self.data[_type].keys()))
        _img_id = random.randint(0,len(self.data[_type][_group])-1)
        
        img, target = self.read_key(_type, _group, _img_id)
        
        
        if self.hard_neg_data is not None:
            hard_neg = self.hard_neg_data[_type][_group][_img_id]
            
            assert (hard_neg['labels'] <= 0).all()
            
            target['coordinates'] = np.concatenate((target['coordinates'], hard_neg['coordinates']))
            target['labels'] = np.concatenate((target['labels'], hard_neg['labels']))
        
        img, target = self.transforms_random(img, target)
        #assert img.shape[-2:] == (self.roi_size, self.roi_size)
        
        return img, target
        

def coords2mask(coords, roi_size, kernel_size = (25,25), sigma = 4):
    b = np.zeros(roi_size, np.float32)
    
    norm_factor = 2*np.pi*(sigma**2) if sigma > 0 else 1.
    
    c_cols, c_rows = np.round(coords).astype(np.int)
    c_rows = np.clip(c_rows, 0, roi_size[0]-1)
    c_cols = np.clip(c_cols, 0, roi_size[1]-1)
    b[c_rows, c_cols]  = norm_factor
    
    if sigma > 0:
        b = cv2.GaussianBlur(b, kernel_size, sigma, sigma, borderType = cv2.BORDER_CONSTANT)  
    return b

#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt
    
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/rois/20x/train'
    #root_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/20x/train'
    #root_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/no_membrane/train'
    #root_dir = Path.home() / 'workspace/localization/data/heba/data-uncorrected/train'   
    
    #root_dir = Path.home() / 'OneDrive - Nexus365/heba/WoundHealing/data4train/mix/train'
    #root_dir = Path.home() / 'workspace/localization/test_images/'
    #loc_gauss_sigma = 2
    
#    root_dir = Path.home() / 'workspace/localization/data/woundhealing/demixed_predictions'
#    loc_gauss_sigma = 2
#    roi_size = 48
    
    #root_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam/validation'
    #root_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam_refined/validation'
    flow_args = dict(
                roi_size = 96,
                #scale_int = (0, 4095),
                scale_int = (0, 255.),
                prob_unseeded_patch = 0.0,
              
                zoom_range = (0.97, 1.03),
                
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.7, 1.3)
                )


    gen = CoordFlow(root_dir, **flow_args)
    
    num_workers = 4
    batch_size = 16
    gauss_sigma = 2
    
    for _ in tqdm.tqdm(range(10)):
        X, target = gen[0]
        
        X = X.numpy()
        if X.shape[0] == 3:
            x = np.rollaxis(X, 0, 3)
        else:
            x = X[0]
        
        plt.figure()
        plt.imshow(x,  cmap='gray', vmin = 0.0, vmax = 1.0)
        
        coords = target['coordinates'].values
        labels = target['labels'].values
        
        for lab in np.unique(labels):
            good = labels == lab
            plt.plot(coords[good, 0], coords[good, 1], 'or')
        
