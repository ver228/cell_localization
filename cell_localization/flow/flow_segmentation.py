#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:27:27 2019

@author: avelinojaver
"""

import math
import tables
import random
from pathlib import Path
import numpy as np
from numpy.lib import recfunctions as rfn
import tqdm

from torch.utils.data import Dataset 

from transforms import ( RandomCropWithSeeds, AffineTransform, RemovePadding, RandomVerticalFlip, 
RandomHorizontalFlip, NormalizeIntensity, RandomIntensityOffset, RandomIntensityExpansion, 
FixDTypes, ToTensor, Compose, OutContours2Segmask )
               
def collate_simple(batch):
    return tuple(map(list, zip(*batch)))

#%%
class FlowCellSegmentation(Dataset):
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
                 prob_unseeded_patch = 0.2,
                 int_aug_offset = None,
                 int_aug_expansion = None,
                 valid_labels = None, # if this is None it will include all the available labels
                 
                 is_preloaded = False,
                 
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
        self.prob_unseeded_patch = prob_unseeded_patch
        self.int_aug_offset = int_aug_offset
        self.int_aug_expansion = int_aug_expansion
        self.valid_labels = valid_labels
        self.is_preloaded = is_preloaded
        self.max_input_size = max_input_size
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        
        
        rotation_pad_size = math.ceil(self.roi_size*(math.sqrt(2)-1)/2)
        padded_roi_size = roi_size + 2*rotation_pad_size
        
        transforms_random = [
                RandomCropWithSeeds(padded_roi_size, rotation_pad_size, prob_unseeded_patch),
                AffineTransform(zoom_range),
                RemovePadding(rotation_pad_size),
                RandomVerticalFlip(),
                RandomHorizontalFlip(),
                NormalizeIntensity(scale_int, norm_mean, norm_sd),
                RandomIntensityOffset(int_aug_offset),
                RandomIntensityExpansion(int_aug_expansion),
                OutContours2Segmask(),
                FixDTypes()
                #I cannot really pass the ToTensor to the dataloader since it crashes when the batchsize is large (>256)
                ]
        self.transforms_random = Compose(transforms_random)
        
        transforms_full = [
                NormalizeIntensity(scale_int),
                OutContours2Segmask(),
                FixDTypes(),
                ToTensor()
                ]
        self.transforms_full = Compose(transforms_full)
        self.hard_neg_data = None
        
        
        if self.data_src.is_dir():
            assert self.folds2include is None
            self.data = self.load_data_from_dir(self.data_src, padded_roi_size, self.is_preloaded)
        else:
            assert self.is_preloaded
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
        
            
        
        with tables.File(str(src_file), 'r') as fid:
            images = fid.get_node('/images')
             
            centroids = self._read_coords(fid, _node = '/localizations')
            contours = self._read_contours(fid, _node = '/contours')
            
            type_ids = np.unique(centroids['type_id'])
            
            data = {tid : {} for tid in type_ids}
            
            keys = centroids['file_id']
            centroids_by_file =  {key: centroids[keys == key] for key in np.unique(keys)}
            
            for file_id, dat in centroids_by_file.items():
                if not _is_valid_fold(file_id):
                    continue
                
                
                img = images[file_id - 1]
                
                if np.all([x <= self.max_input_size for x in img.shape[:-1]]):
                
                    for tid in np.unique(dat['type_id']):
                        data[tid][file_id] = [(img, dat, contours[file_id])]
                else:
                    
                    #divide files if they are too large...
                    inds = []
                    for ss in img.shape[:-1]:
                        n_split = int(np.ceil(ss/self.max_input_size)) + 1
                        ind = np.linspace(0, ss, n_split)[1:-1].round().astype(np.int)
                        inds.append([0, *ind.tolist(), ss])
                    
                    inds_i, inds_j = inds
                    for i1, i2 in zip(inds_i[:-1], inds_i[1:]):
                        for j1, j2 in zip(inds_j[:-1], inds_j[1:]):
                            roi = img[i1:i2, j1:j2]
                            
                            valid = (dat['cy'] >=  i1) & (dat['cy'] <  i2) & (dat['cx'] >=  j1) & (dat['cx'] <  j2)
                            roi_dat = dat[valid].copy()
                            
                            roi_dat['cy'] -= i1
                            roi_dat['cx'] -= j1
                            
                            x2add = (roi, roi_dat)
                            for tid in np.unique(roi_dat['type_id']):
                                if not file_id in data[tid]:
                                    data[tid][file_id] = []
                                data[tid][file_id].append(x2add)
                                
                    
        return data
                
    def load_data_from_dir(self, root_dir, padded_roi_size, is_preloaded = True):
        data = {} 
        fnames = [x for x in root_dir.rglob('*.hdf5') if not x.name.startswith('.')]
        
        header = 'Preloading Data' if is_preloaded else 'Reading Data'
        for ifname, fname in enumerate(tqdm.tqdm(fnames, header)):
            with tables.File(str(fname), 'r') as fid:
                
                img = fid.get_node('/img')
                img_shape = img.shape[:2]
                
                if any([x < padded_roi_size for x in img_shape]):
                    import pdb
                    pdb.set_trace()
                    continue
                
                if not '/coords' in fid:
                    continue
                
                coords = self._read_coords(fid)
                contours = self._read_contours(fid)
                
                if is_preloaded:
                    x2add = (img[:], coords, contours)
                else:
                    x2add = fname
            
            type_ids = set(np.unique(coords['type_id']).tolist())
            
            
            k = fname.parent.name
            for tid in type_ids:
                if tid not in data:
                    data[tid] = {}
                if k not in data[tid]:
                    data[tid][k] = []
                
                data[tid][k].append(x2add)
        
        return data
    
    def _read_coords(self, fid, _node = '/coords'):
        rec = fid.get_node(_node)[:]
        if not 'type_id' in rec.dtype.names:
            type_id = np.ones(len(rec), dtype = np.int32)
            rec = rfn.append_fields(rec, 'type_id', type_id)
        
        if self.valid_labels is not None:
            valid = np.isin(rec['type_id'], self.valid_labels)
            rec = rec[valid]
        return rec

    def _read_contours(self, fid, _node = '/contours'):
        
        
        if not _node in fid:
            return None
        
        def _rec2array(cnt):
            return np.stack((cnt['x'], cnt['y'])).T
        
        def _groupbynuclei(dat):
            nuclei_ids = dat['nuclei_id']
            return {key: _rec2array(dat[nuclei_ids == key]) for key in np.unique(nuclei_ids)}
        
        contours = fid.get_node(_node)[:]
        if 'file_id' in contours.dtype.names:
            file_ids = contours['file_id']
            return {key: _groupbynuclei(contours[file_ids == key]) for key in np.unique(file_ids)}
            
        else:
            return _groupbynuclei(contours)
        
        
    def __len__(self):
        return self.samples_per_epoch
            
    
    def __getitem__(self, ind):
        img, target = self.read_random()
        return img, target
        
        

    def read_full(self, ind):
        (_type, _group, _img_id) = self.data_indexes[ind]
        img, target = self.read_key(_type, _group, _img_id)
        img, target = self.transforms_full(img, target)
        
        return img, target
    
    def read_key(self, _type, _group, _img_id):
        input_ = self.data[_type][_group][_img_id]
        
        if self.is_preloaded:
           img, coords_rec, contours_dict = input_
        else:
            fname = input_
            with tables.File(str(fname), 'r') as fid:
                coords_rec = self._read_coords(fid)
                contours_dict = self._read_contours(fid)
                img = fid.get_node('/img')[:]
                
        
        
        labels = np.array([self.types2label[x] for x in coords_rec['type_id']])
        target = dict(
                labels = labels,
                coordinates = np.array((coords_rec['cx'], coords_rec['cy'])).T
                )
        
        if contours_dict is not None:
            target['contours'] = [contours_dict[x] for x in coords_rec['nuclei_id']]
            
        
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


    
#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt

    src_file = ''
    
    #root_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/all_lymphocytes/20x/validation/'
    #root_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/splitted/F0.5x/'
    
    #root_dir = Path.home() / 'workspace/localization/data/histology_data/40x_MoNuSeg_training.hdf5'
    
    #root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/reformated/40x_MoNuSeg_training.hdf5'
    #root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/reformated/40x_MoNuSeg_training.hdf5'
    
    #root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/reformated/40x_TMA_lymphocytes_2Bfirst104868.hdf5'
    #root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/reformated/20x-resized_TMA_lymphocytes_2Bfirst104868.hdf5'
    #root_dir = '/Users/avelinojaver/Desktop/nuclei_datasets/reformated/40x_andrewjanowczyk_lymphocytes.hdf5'
    #root_dir = Path.home() / 'workspace/localization/data/histology_data/40x_andrewjanowczyk_lymphocytes.hdf5'
    
    root_dir =  '/Users/avelinojaver/Desktop/nuclei_datasets/separated_files/BBBC038_Kaggle_2018_Data_Science_Bowl/fluorescence/train/'
    #root_dir =  '/Users/avelinojaver/Desktop/nuclei_datasets/separated_files/BBBC038_Kaggle_2018_Data_Science_Bowl/fluorescence/validation/'
    
    flow_args = dict(
                roi_size = 96,
                #scale_int = (0, 4095),
                scale_int = (0, 255.),
                #scale_int = (0, 1.),
                prob_unseeded_patch = 0.0,
              
                zoom_range = (0.90, 1.1),
                
                int_aug_offset = None,#(-0.2, 0.2),
                int_aug_expansion = None, #(0.5, 1.3),
                valid_labels = [1],
                is_preloaded = True,
                
                #folds2include = 1#[1,2,3,4]
                )


    gen = FlowCellSegmentation(root_dir, **flow_args)
#    #import pyximport; pyximport.install()
#    from clip_contours import crop_contour
#    cnt = gen.data[1][1][0][2][1].astype(np.int)
#    
#    xl, xr, yl, yr = 90,  160,  60,  90
#    
#    cnt_o = crop_contour(cnt,  xl, xr, yl, yr)
#    plt.figure()
#    plt.plot(cnt[:, 0] - xl, cnt[:, 1] - yl)
#    plt.plot(cnt_o[:, 0], cnt_o[:, 1])
#%%
    col_dict = {1 : 'r', 2 : 'g'}
    for _ in tqdm.tqdm(range(10)):
        X, target = gen[0]
    
        
        #X = X.numpy()
        if X.shape[0] == 3:
            #x = X[::-1]
            x = X
            x = np.rollaxis(x, 0, 3)
        else:
            x = X[0]
        
        fig, axs = plt.subplots(1, 2, sharex = True, sharey = True)
        axs[0].imshow(x,  cmap='gray', vmin = 0.0, vmax = 1.0)
        
        if 'segmentation_mask' in target:
            seg = target['segmentation_mask']
            axs[1].imshow(seg)
        else:
            axs[1].imshow(x,  cmap='gray', vmin = 0.0, vmax = 1.0)
            if 'coordinates' in target:
                coords = target['coordinates']
                labels = target['labels']
                assert (labels > 0).all()
                
                
                for lab in np.unique(labels):
                    good = labels == lab
                    axs[1].plot(coords[good, 0], coords[good, 1], 'o', color = col_dict[lab])
                
            if 'contours' in target:
                for cnt in target['contours']:
                    axs[1].plot(cnt[:, 0], cnt[:, 1], 'r')
        
