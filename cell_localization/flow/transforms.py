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
from clip_contours import c_crop_contour
#def _clip_to_crop(p1, p2, bad, val):
#    p1[bad] = val
#    vv = p2[~bad]
#    if not vv.size:
#        return np.zeros((0, 2))
#    
#    p2[bad] = np.clip(p2[bad], vv.min(), vv.max())
#    return p1, p2
#
#def crop_contour(coords, xl, xr, yl, yr):
#    xx, yy = coords.T
#    
#    xx = coords[:, 0] - xl
#    yy = coords[:, 1] - yl
#    
#    xlim = xr - xl - 1
#    ylim = yr - yl - 1
#    
#    is_xmin = xx < 0
#    is_xmax = xx >= xlim
#    is_ymin = yy < 0
#    is_ymax = yy >= ylim
#    
#    
#    if np.all(is_xmin | is_xmax | is_ymin | is_ymax):
#        return np.zeros((0, 2))
#    else:
#        _clip_to_crop(xx, yy, is_xmin, 0)
#        _clip_to_crop(xx, yy, is_xmax, xlim)
#        _clip_to_crop(yy, xx, is_ymin, 0)
#        _clip_to_crop(yy, xx, is_ymax, ylim)
#        
#        return np.stack((xx, yy)).T

def crop_contour(coords, xl, xr, yl, yr):
    return c_crop_contour(coords, xl, xr, yl, yr)
    


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class TransformBase(object):
    def _apply_transform(self, image, target, *args, **argkws):
        image = self._from_image(image, *args, **argkws)
        
        if 'coordinates' in target:
            target['coordinates'], target['labels'] = self._from_coordinates(target['coordinates'], *args, labels = target['labels'], **argkws)
            assert len(target['coordinates']) == len(target['labels'])
        
        if 'contours' in target:
             contours = [self._from_contours(c, *args, **argkws) for c in target['contours']]
             target['contours'] = [x for x in contours if len(x) > 0]
        
        if 'mask' in target:
            target['mask'] = self._from_image(target['mask'], *args, **argkws)
            
        return image, target
    
    def _from_coordinates(self, coordinates, *args, labels = None, **argkws):
        if labels is not None:
            return coordinates, labels
        else:
            return coordinates
    
    def _from_contours(self,  *args, **argkws):
        return self._from_coordinates(*args, **argkws)
    
    def _from_image(self, image, *args, **argkws):
        return image
    
class RandomCrop(TransformBase):
    def __init__(self, crop_size = 512):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        
        self.crop_size = crop_size
    
    
    def __call__(self, image, target):
        #### select the limits allowed for a random crop
        xlims = (0, max(1, image.shape[1] - self.crop_size[0]))
        ylims = (0, max(1, image.shape[0] - self.crop_size[1]))
        
        #### crop with padding in order to keep a valid rotation 
        xl = random.randint(*xlims)
        yl = random.randint(*ylims)
        
        yr = yl + self.crop_size[0]
        xr = xl + self.crop_size[1]
        return self._apply_transform(image, target, xl, xr, yl, yr)
        
    def _from_coordinates(self, coords, xl, xr, yl, yr, labels = None):
        xx, yy = coords.T
        valid = (xx > xl) & (xx < xr)
        valid &= (yy > yl) & (yy < yr)
        
        coord_out = coords[valid].copy()
        coord_out[:, 0] -= xl
        coord_out[:, 1] -= yl
        
        
        if labels is None:
            return coord_out
        else:
            return coord_out, labels[valid].copy()
    
    def _from_contours(self, coords, xl, xr, yl, yr):
        
        return crop_contour(coords, xl, xr, yl, yr)
    
    def _from_image(self, image, xl, xr, yl, yr):
        return image[yl:yr, xl:xr].copy()


class RandomCropWithSeeds(RandomCrop):
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
    
        return self._apply_transform(image, target, xl, xr, yl, yr)
        
#%%

class AffineTransform(TransformBase):
    def __init__(self, 
                 zoom_range = (0.9, 1.1), 
                 rotation_range = (-90, 90), 
                 border_mode = cv2.BORDER_CONSTANT
                 ):
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
        self.w_affine_args = dict(borderMode = border_mode)
    
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
        
        return self._apply_transform(image, target, M)
    
    
    def _from_coordinates(self, coords, M, img_shape = None, labels = None):
        coords = np.dot(M[:2, :2], coords.T)  +  M[:, -1][:, None]
        coords = coords.T #np.round(coords).T
        
        if labels is None:
            return coords
        else:
            return coords, labels
        
    def _from_image(self, image, M, img_shape = None):
        img_shape = image.shape[:2] if img_shape is None else img_shape
        if image.ndim == 2:
            image_rot = cv2.warpAffine(image, M, img_shape, **self.w_affine_args)
        else:
            image_rot = [cv2.warpAffine(image[..., n], M, img_shape, **self.w_affine_args) for n in range(image.shape[-1])] 
            image_rot = np.stack(image_rot, axis=2)
        return image_rot
        

class RemovePadding(TransformBase):
    def __init__(self, pad_size):
        self.pad_size = pad_size
    
    def __call__(self, image, target):
        
        expected_size = tuple([x - 2*self.pad_size for x in image.shape[:2]])
        return self._apply_transform(image, target, expected_size)
 
    
    def _from_coordinates(self, coords, expected_size, labels = None):
        coord_out = coords - self.pad_size
        valid = (coord_out[:, 0] >= 0) & (coord_out[:, 0] < expected_size[1])
        valid &= (coord_out[:, 1] >= 0) & (coord_out[:, 1] < expected_size[0])
        coord_out = coord_out[valid].copy()
        
        if labels is None:
            return coord_out
        else:
            return coord_out, labels[valid].copy()
        
        
    def _from_contours(self, coords, expected_size):
        xl = yl = self.pad_size
        xr = expected_size[0] + self.pad_size
        yr = expected_size[1] + self.pad_size
        return crop_contour(coords, xl, xr, yl, yr)
    
    def _from_image(self, image, expected_size):
        img_cropped = image[self.pad_size:-self.pad_size, self.pad_size:-self.pad_size].copy()
        assert img_cropped.shape[:2] == expected_size
        return img_cropped

class RandomVerticalFlip(TransformBase):
    def __init__(self, prob = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            return self._apply_transform(image, target, image.shape)
            
        return image, target
    
    @staticmethod
    def _from_coordinates(coords, img_shape, labels = None):
        coords_out = coords.copy()
        coords_out[:, 1] = (img_shape[0] - 1 - coords_out[:, 1])
        
        if labels is None:
            return coords_out
        else:
            return coords_out, labels
        
    
    @staticmethod
    def _from_image(image, *args, **argkws):
        return image[::-1]
    
    
class RandomHorizontalFlip(TransformBase):
    def __init__(self, prob = 0.5):
        self.prob = prob
    
    def __call__(self, image, target):
       
        if random.random() < self.prob:
            return self._apply_transform(image, target, image.shape)
        
        return image, target
    
    
    @staticmethod
    def _from_coordinates(coords, img_shape, labels = None):
        coords_out = coords.copy()
        coords_out[:, 0] = (img_shape[1] - 1 - coords_out[:, 0])
        
        if labels is None:
            return coords_out
        else:
            return coords_out, labels
    
    
    @staticmethod
    def _from_image(image, *args, **argkws):
        return image[:, ::-1]

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
                
            image = image*factor
            
        return image, target    

class FixDTypes(TransformBase):
    def __call__(self, image, target):
        
        image, target = self._apply_transform(image, target)
        
        #it is a bit annoying to assing the type in a single function (_from_image) so i will do it after
        image = image.astype(np.float32)
        if 'mask' in target:
           target['mask'] = target['mask'].astype(np.int)
        
        return image, target
    
    @staticmethod
    def _from_coordinates(coords, labels = None):
        coords = coords.astype(np.int)
        
        if labels is None:
            return coords
        else:
            labels = labels.astype(np.int)
            return coords, labels
        
    @staticmethod
    def _from_image(image):
        if image.ndim == 3:
            image = np.rollaxis(image, 2, 0).copy()
        else:
            image = image[None]
        
        return image
        
class ToTensor(object):
    def __call__(self, image, target):
        image = torch.from_numpy(image).float()
        
        if 'coordinates' in target:
            target['coordinates'] = torch.from_numpy(target['coordinates']).long()
            target['labels'] = torch.from_numpy(target['labels']).long()
            
        if 'contours' in target:
            target['contours'] = [torch.from_numpy(x).long() for x in  target['contours']]
        
        if 'mask' in target:
            target['mask'] = torch.from_numpy(target['mask']).long()
            
        if 'segmentation_mask' in target:
            target['segmentation_mask'] = torch.from_numpy(target['segmentation_mask']).long()
            
        return image, target
        

class OutContours2Segmask(object):
    def __call__(self, image, target):
    
        contours = target['contours']
        contours_i = [np.floor(x).astype(np.int) for x in contours]
        
        seg_mask = np.zeros(image.shape[:2], dtype = np.uint8)
        
        contours_i = sorted(contours_i, key = cv2.contourArea)
        for cnt in contours_i:
            cv2.drawContours(seg_mask, [cnt], 0, 1, -1)
            
        for cnt in contours_i:
            #rect = cv2.minAreaRect(cnt)
            #max_length = min(rect[1])
            #thickness = max(1, int(np.log10(max_length + 1) + 1) )
            thickness = 1
            cv2.drawContours(seg_mask, [cnt], 0, 2, thickness)
        
        target_out = dict(segmentation_mask = seg_mask.astype(np.int))
        
        return image, target_out
    