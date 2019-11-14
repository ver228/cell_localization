#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 21:49:29 2019

@author: avelinojaver
"""
import numpy as np
import random
import torch
import cv2

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
        if 'coordinates' in target and random.random() >= self.prob_unseeded:
            
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
        
        if 'coordinates' in target:
            xx, yy = target['coordinates'].T
            valid = (xx > xl) & (xx < xr)
            valid &= (yy > yl) & (yy < yr)
            
            #I might include labels <= 0 to indicate coordinate from hard negative mining. So let's remove it here.
            valid &= target['labels'] > 0
            
            coordinates = target['coordinates'][valid].copy()
            coordinates[:, 0] -= xl
            coordinates[:, 1] -= yl
            
            target['coordinates'] = coordinates
            target['labels'] = target['labels'][valid].copy()
        
        if 'mask' in target:
            target['mask'] = target['mask'][yl:yr, xl:xr].copy()
        
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
        
        if 'coordinates' in target:
            coords = target['coordinates'].T
            coords_rot = np.dot(M[:2, :2], coords)  +  M[:, -1][:, None]
            coords_rot = np.round(coords_rot).T
            
            target['coordinates'] = coords_rot
            
        if 'mask' in target:
            target['mask'] = cv2.warpAffine(target['mask'], M, (rows, cols))
            
        
        return image_rot, target

class RemovePadding():
    def __init__(self, pad_size):
        self.pad_size = pad_size
    
    def __call__(self, image, target):
        ##### remove padding
        img_out = image[self.pad_size:-self.pad_size, self.pad_size:-self.pad_size].copy()
        
        if 'coordinates' in target:
            coord_out = target['coordinates'] - self.pad_size
            
            valid = (coord_out[:, 0] >= 0) & (coord_out[:, 0] < img_out.shape[1])
            valid &= (coord_out[:, 1] >= 0) & (coord_out[:, 1] < img_out.shape[0])
            coord_out = coord_out[valid].copy()
            
            target['coordinates'] = coord_out
            target['labels'] = target['labels'][valid].copy()
            
        if 'mask' in target:
            target['mask'] = target['mask'][self.pad_size:-self.pad_size, self.pad_size:-self.pad_size].copy()
        
        return img_out, target
    
class RandomVerticalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[::-1]
        
            if 'coordinates' in target:
                coordinates = target['coordinates'].copy()
                coordinates[:, 1] = (image.shape[0] - 1 - coordinates[:, 1])
                target['coordinates'] = coordinates
            
            if 'mask' in target:
                target['mask'] = target['mask'][::-1]
        
        
        return image, target
    
class RandomHorizontalFlip(object):
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    def __call__(self, image, target):
        if random.random() < self.prob:
            image = image[:, ::-1]
        
            if 'coordinates' in target:
                coordinates = target['coordinates'].copy()
                coordinates[:, 0] = (image.shape[1] - 1 - coordinates[:, 0])
                target['coordinates'] = coordinates
            
            if 'mask' in target:
                target['mask'] = target['mask'][:, ::-1]
        
        return image, target

class NormalizeIntensity(object):
    def __init__(self, scale = (0, 255), norm_mean = 0., norm_sd = 1.):
        self.scale = scale
        self.norm_mean = norm_mean
        self.norm_sd = norm_sd
        
    def __call__(self, image, target):
        
        image = (image.astype(np.float32) - self.scale[0])/(self.scale[1] - self.scale[0])
        
        return image, target
        

class RandomIntensityOffset(object):
    def __init__(self, offset_range = (-0.2, 0.2)):
        self.offset_range = offset_range
    
    def __call__(self, image, target):
        if self.offset_range is not None and random.random() > 0.5:
            
            if image.ndim == 2:
                offset = random.uniform(*self.offset_range)
            else:
                offset = np.random.uniform(*self.offset_range, 3)[None, None, :]
            image = image + offset
                
        return image, target


class RandomIntensityExpansion(object):
    def __init__(self, expansion_range = (0.7, 1.3)):
        self.expansion_range = expansion_range
    
    def __call__(self, image, target):
        if self.expansion_range is not None and random.random() > 0.5:
            if image.ndim == 2:
                factor = random.uniform(*self.expansion_range)
            else:
                factor = np.random.uniform(*self.expansion_range, 3)[None, None, :]
                import pdb
                pdb.set_trace()
            image = image*factor
            
        return image, target    

class FixDTypes(object):
    def __call__(self, image, target):
        if image.ndim == 3:
            image = np.rollaxis(image, 2, 0).copy()
        else:
            image = image[None]
        
        image = image.astype(np.float32)
        
        if 'coordinates' in target:
            target['coordinates'] = target['coordinates'].astype(np.int)
            target['labels'] = target['labels'].astype(np.int)
        
        if 'mask' in target:
            mask = target['mask']
            if mask.ndim == 3:
                mask = np.rollaxis(mask, 2, 0).copy()
            else:
                mask = mask[None]
            
            target['mask'] = mask.astype(np.int)
        
        return image, target
    
class ToTensor(object):
    def __call__(self, image, target):
        
        image = torch.from_numpy(image).float()
        
        if 'coordinates' in target:
            target['coordinates'] = torch.from_numpy(target['coordinates']).long()
            target['labels'] = torch.from_numpy(target['labels']).long()
        
        if 'mask' in target:
            target['mask'] = torch.from_numpy(target['mask']).float()
        
        return image, target