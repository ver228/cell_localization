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
from torch.utils.data import Dataset 
import numpy.lib.recfunctions as rfn


#%%
class CoordFlow(Dataset):
    def __init__(self, 
                 root_dir,
                 samples_per_epoch = 2000,
                 roi_size = 96,
                 zoom_range = (0.90, 1.1),
                 scale_int = (0, 255),
                 loc_gauss_sigma = 3,
                 bbox_encoder = None,
                 min_radius = 2.5,
                 prob_unseeded_patch = 0.2,
                 int_aug_offset = None,
                 int_aug_expansion = None,
                 patchnorm = False,
                 is_preloaded = False,
                 ignore_borders = False,
                 stack_shape = None
                 ):
        
        
        self.root_dir = Path(root_dir)
        self.samples_per_epoch = samples_per_epoch
        
        self.roi_size = roi_size
        self.roi_padding = math.ceil(roi_size*(math.sqrt(2)-1)/2)
        self.padded_roi_size = self.roi_size + 2*self.roi_padding
        self.loc_gauss_sigma = loc_gauss_sigma
        
        self.zoom_range = zoom_range
        self.scale_int = scale_int
        self.bbox_encoder = bbox_encoder
        self.min_radius = min_radius
        self.prob_unseeded_patch = prob_unseeded_patch
        
        self.int_aug_offset = int_aug_offset
        self.int_aug_expansion = int_aug_expansion
        
        self.patchnorm = patchnorm
        
        self.is_preloaded = is_preloaded
        self.ignore_borders = ignore_borders
        self.stack_shape = stack_shape
        
        
        dat = {}
        fnames = [x for x in self.root_dir.rglob('*.hdf5') if not x.name.startswith('.')]
        for fname in fnames:
            with tables.File(str(fname), 'r') as fid:
                img = fid.get_node('/img')
                img_shape = img.shape[:2]
                
                if any([x < self.padded_roi_size for x in img_shape]):
                    continue
                
                if not '/coords' in fid:
                    continue
                
                rec = fid.get_node('/coords')[:]
                type_ids = set(np.unique(rec['type_id']).tolist())
                
                
                if self.is_preloaded:
                    rec = self._add_radius(rec)
                    x2add = (img[:], rec)
                else:
                    x2add = fname
                
            k = fname.parent.name
            
            for tid in type_ids:
                if tid not in dat:
                    dat[tid] = {}
                
                if k not in dat[tid]:
                    dat[tid][k] = []
                
                dat[tid][k].append(x2add)
            
        self.slides_data = dat
        self.type_ids = sorted(list(dat.keys()))
        
        self.slide_data_indexes = [(_type, _slide, ii)for _type, type_data in self.slides_data.items() 
                                    for _slide, slide_data in type_data.items() for ii in range(len(slide_data))]
        
        self.hard_neg_data = None
        
        
    def __len__(self):
        return self.samples_per_epoch
            
    
    def __getitem__(self, ind):
        
        def _read():
            roi, coords_rec = self.read_random()
            target = self._prepare_target(coords_rec)
            return roi.astype(np.float32), target
            
        if self.stack_shape is None:
            return _read()
            
        else:
            rows = []
            for _ in range(self.stack_shape[0]):
                cols = []
                for _ in range(self.stack_shape[1]):
                    cols.append(_read())
                    
                cols = [np.concatenate(x, axis=1) for x in zip(*cols)]
                
                rows.append(cols)
              
            out = [np.concatenate(x, axis=2) for x in zip(*rows)]
        
            return out
            
    
    def read_full(self, ind):
        (_type, _slide, _img_ind) = self.slide_data_indexes[ind]
        data = self.slides_data[_type][_slide][_img_ind]
        
        if self.is_preloaded:
           img, coords_rec = data
        else:
            _file = data
            with tables.File(str(_file), 'r') as fid:
                img = fid.get_node('/img')[:]
                coords_rec = fid.get_node('/coords')[:]
                coords_rec = self._add_radius(coords_rec)
        img, coords_rec = self._prepare_data(img, coords_rec, is_augment = False)
        
        return img, coords_rec
    
    def read_random(self):
        _type = random.choice(self.type_ids)
        _slide = random.choice(list(self.slides_data[_type].keys()))
        _exp_id = random.randint(0,len(self.slides_data[_type][_slide])-1)
        
        data = self.slides_data[_type][_slide][_exp_id]
        
        hard_neg_rec = None
        if self.hard_neg_data is not None:
            hard_neg_rec = self.hard_neg_data[_type][_slide][_exp_id]
        
        if self.is_preloaded:
           img, coords_rec = data
           
           roi, roi_coords_rec = self._prepare_data(img, coords_rec, hard_neg_rec = hard_neg_rec)
        else:
            _file = data
            with tables.File(str(_file), 'r') as fid:
                coords_rec = fid.get_node('/coords')[:]
                coords_rec = self._add_radius(coords_rec)
                
                #here the node is open but i will not load the data until i select the correct ROI
                img = fid.get_node('/img')
                roi, roi_coords_rec = self._prepare_data(img, coords_rec, hard_neg_rec = hard_neg_rec)
                
        return roi, roi_coords_rec
    
    def _prepare_target(self, coords_rec):
        if self.bbox_encoder is None:
            #return coordinates maps
            if coords_rec.size > 0:
                #use a different mask per channel
                coords_mask = []
                
                roi_shape = (self.roi_size, self.roi_size)
                for type_id in self.type_ids:
                    
                    good = coords_rec['type_id'] == type_id
                    
                    cc = coords_rec[good]
                    xys = np.array((cc['cx'], cc['cy']))
                    
                    mm = coords2mask(xys, roi_shape, sigma = self.loc_gauss_sigma)    
                    coords_mask.append(mm)
                coords_mask = np.array(coords_mask)
                
            else:
                coords_mask = np.zeros((len(self.type_ids), self.roi_size, self.roi_size))
    
            target = coords_mask.astype(np.float32)
            
        else:
            labels = coords_rec['type_id'].astype(np.int)
            
            #x1,y1, x2, y2
            rr = coords_rec['radius']
            x1 = coords_rec['cx'] - rr
            x2 = coords_rec['cx'] + rr
            y1 = coords_rec['cy'] - rr
            y2 = coords_rec['cy'] + rr
            
            bboxes = np.stack((x1,y1, x2, y2)).T
            bboxes = bboxes if bboxes.ndim == 2 else bboxes[None]
            
            clf_target, loc_target = self.bbox_encoder.encode(labels, bboxes)
            
            target = clf_target.astype(np.int), loc_target.astype(np.float32)
            
        return target
    
    
    def _add_radius(self, coords_rec):
        try:
            coords_rec['radius']
        except ValueError:
            rr = np.full(len(coords_rec), float(self.min_radius))
            coords_rec = rfn.append_fields(coords_rec, 'radius', rr)
        return coords_rec
    
    
    def _get_aug_seed(self, coords_rec, img_dims, hard_neg_rec = None):
        seed_row = None
        if random.random() > self.prob_unseeded_patch:
            
            if hard_neg_rec is None or hard_neg_rec.size == 0 or random.random() > 0.5:
                coord2choice = coords_rec
            else:
                coord2choice = hard_neg_rec
                
            
            _type = random.choice(np.unique(coord2choice['type_id']))
            #randomly select a type
            rec = coord2choice[coord2choice['type_id'] == _type]
            
            
            yl, xl = img_dims
            x, y, r = rec['cx'], rec['cy'], rec['radius']
            
            good = (x >= self.roi_padding + r ) & (x  < (xl - self.roi_padding - r))
            good &= (y >= self.roi_padding + r ) & (y < (yl - self.roi_padding - r ))
            rec = rec[good]
            
            if len(rec) > 0:
                seed_row = [int(x) for x in random.choice(rec[['cx', 'cy', 'radius']])]
        return seed_row
            
    
    def _prepare_data(self, img, coords_rec, is_augment = True, hard_neg_rec = None):
        if is_augment:
            ### I either randomly selecting a roi, 
            # or making sure the roi includes a randomly selected labeled point.
            seed_row = self._get_aug_seed(coords_rec, img.shape[:2], hard_neg_rec = hard_neg_rec)
            img, coords_rec = self._crop_augment(img, coords_rec, seed_row)
        
        if not self.patchnorm:
            img = (img.astype(np.float32) - self.scale_int[0])/(self.scale_int[1] - self.scale_int[0])
        else:
            img = img.astype(np.float32)
            img -= img.mean()
            img /= img.std()
            
        
        if img.ndim == 3:
            ### channel first for pytorch compatibility
            img = np.rollaxis(img, 2, 0)
        else:
            img = img[None]
        
        if is_augment:
            if self.int_aug_offset is not None and random.random() > 0.5:
                int_offset = random.uniform(*self.int_aug_offset)
                img += int_offset
                
            if self.int_aug_expansion is not None and random.random() > 0.5:
                int_expansion = random.uniform(*self.int_aug_expansion)
                img *= int_expansion
        
        
        return img, coords_rec


    def _crop_augment(self, img, coord_rec, seed_row = None):
        #### select the limits allowed for a random crop
        xlims = (self.roi_padding, img.shape[1] - self.roi_size - self.roi_padding - 1)
        ylims = (self.roi_padding, img.shape[0] - self.roi_size - self.roi_padding - 1)
            
        if seed_row is not None:
            p = self.roi_size
            x,y,r = seed_row
            
            seed_xlims = (x + r - p, x - r)
            seed_xlims = list(map(int, seed_xlims))
            
            seed_ylims = (y + r - p, y - r)
            seed_ylims = list(map(int, seed_ylims))
            
            x1 = max(xlims[0], seed_xlims[0])
            x2 = min(xlims[1], seed_xlims[1])
            
            y1 = max(ylims[0], seed_ylims[0])
            y2 = min(ylims[1], seed_ylims[1])
            
            
            xlims = x1, x2
            ylims = y1, y2
            
        #### crop with padding in order to keep a valid rotation 
        xl = random.randint(*xlims) - self.roi_padding
        yl = random.randint(*ylims) - self.roi_padding
        
        yr = yl + self.padded_roi_size
        xr = xl + self.padded_roi_size
    
        crop_padded = img[yl:yr, xl:xr]
        
        if crop_padded.shape[:2] != (self.padded_roi_size, self.padded_roi_size):
            #import pdb
            #pdb.set_trace()
            raise ValueError(f'Incorrect crop size {crop_padded.shape[:2]}. This needs to be debugged.')
            
        
        
        
        valid_coords = (coord_rec['cx']> xl) & (coord_rec['cx']< xr)
        valid_coords &= (coord_rec['cy']> yl) & (coord_rec['cy']< yr)
        
        coord_out = coord_rec[valid_coords]
        coord_out['cx'] -= xl
        coord_out['cy'] -= yl
        
        ##### rotate
        #crop_rotated = crop_padded
        
        theta = np.random.uniform(-180, 180)
        scaling = 1/np.random.uniform(*self.zoom_range)
        
        xys = np.array((coord_out['cx'], coord_out['cy']))
        
        crop_rotated, xys = affine_transform(crop_padded, xys, theta, scaling)
        #important, otherwise it will do the rounding by removing the decimal part cast from float to int
        coord_out['cx'] = np.round(xys[0])
        coord_out['cy'] = np.round(xys[1])
        
        coord_out['radius'] *= scaling
        coord_out['radius'][coord_out['radius'] < self.min_radius] = self.min_radius
        
        
        ##### remove padding
        crop_out = crop_rotated[self.roi_padding:-self.roi_padding, self.roi_padding:-self.roi_padding]
        coord_out['cx'] -= self.roi_padding
        coord_out['cy'] -= self.roi_padding
        
        
        if not self.ignore_borders:
            left_lim, right_lim = 0, self.roi_size
        else:
            ss = self.min_radius
            left_lim, right_lim = ss, self.roi_size - ss
            
        valid_coords = (coord_out['cx']> left_lim) & (coord_out['cx']< right_lim)
        valid_coords &= (coord_out['cy']> left_lim) & (coord_out['cy']< right_lim)
        coord_out = coord_out[valid_coords]
        
        
        
        
        ##### flips
        if random.random() > 0.5:
            crop_out = crop_out[::-1]
            coord_out['cy'] = (crop_out.shape[1] - 1) - coord_out['cy'] 
        
        if random.random() > 0.5:
            crop_out = crop_out[:, ::-1]
            coord_out['cx'] = (crop_out.shape[0] - 1) - coord_out['cx'] 
        
        
        
        if len(coord_out) > 0 :
            assert np.all(coord_out['cx']>=0) and np.all(coord_out['cy']>=0)
        
        
        return crop_out, coord_out
    
    
    
    
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
    
    if img.ndim == 2:
        img = cv2.warpAffine(img, M, (rows, cols))#, borderMode = cv2.BORDER_REFLECT_101)
    else:
        for n in range(img.shape[-1]):
            img[..., n] = cv2.warpAffine(img[..., n], M, (rows, cols), flags = cv2.INTER_CUBIC)
    
    coords_rot = np.dot(M[:2, :2], coords)  +  M[:, -1][:, None]
    
    return img, coords_rot   

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
    from skimage.feature import peak_local_max
    import matplotlib.pylab as plt
    import tqdm
    from torch.utils.data import DataLoader
    
    #root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/rois/20x/train'
#    root_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/20x/train'
#    loc_gauss_sigma = 2.5
#    roi_size = 64
#    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/rois/40x/train'
#    loc_gauss_sigma = 5
#    roi_size = 128

#    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/full_tiles/40x'
#    loc_gauss_sigma = 5
#    roi_size = 64    

#    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/full_tiles/20x'
#    loc_gauss_sigma = 2.5
#    roi_size = 48        

#    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/cell_detection/data/validation'
#    loc_gauss_sigma = 2
#    roi_size = 48

#    root_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/no_membrane/train'
#    #root_dir = Path.home() / 'workspace/localization/data/heba/data-uncorrected/train'   
#    loc_gauss_sigma = 2
#    roi_size = 48
    
#    root_dir = Path.home() / 'workspace/localization/data/woundhealing/demixed_predictions'
#    loc_gauss_sigma = 2
#    roi_size = 48
    
    root_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam/validation'
    #root_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam_refined/validation'
    flow_args = dict(
                roi_size = 48,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.25,
                loc_gauss_sigma = -1,#1.5,
                zoom_range = (0.97, 1.03),
                ignore_borders = False,
                min_radius = 2.,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.7, 1.3),
                stack_shape = (4,4)
                )


    gen = CoordFlow(root_dir,
                     
                     bbox_encoder = None,
                     **flow_args
                     )
    
    num_workers = 4
    batch_size = 16
    loader = DataLoader(gen, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=num_workers)
    
    

    #%%
    for _ in tqdm.tqdm(range(10)):
        X,Y = gen[0]
        #%%
        if X.shape[0] == 3:
            x = np.rollaxis(X, 0, 3)
            x = x[..., ::-1]
        else:
            x = X[0]
        
        fig, axs = plt.subplots(1, Y.shape[0] + 1,sharex=True,sharey=True)#, figsize= (20, 20))
        axs[0].imshow(x, cmap='gray', vmin = 0.0, vmax = 1.0)
        
        ccs = 'rc'
        for ii, y in enumerate(Y):
            coords_pred = peak_local_max(y, min_distance = 2, threshold_abs = 0.1, threshold_rel = 0.5)
            axs[ii + 1].imshow(y)
            axs[0].plot(coords_pred[...,1], coords_pred[...,0], 'x', color = ccs[ii])
        #%%
    plt.show()
