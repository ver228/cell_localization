#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 16:51:08 2018

@author: avelinojaver
"""
from pathlib import Path

from .encoder import BoxEncoder

import json
import pandas as pd
import cv2
import numpy as np
import random
import math

import torch
from torch.utils.data import Dataset 

root_dir = Path.home() / 'workspace/Vesicles'
file_bbox = root_dir / 'training_data/vesicles_locations.json'


def read_bb_attrs(file_bbox):
    with open(str(file_bbox), 'r') as fid:
        via_data = json.load(fid)
    
    valid_data = []
    for img_name, metadata in via_data['_via_img_metadata'].items():
        img_id = img_name.split('.')[0]
        
        for rr in metadata['regions']:
            attrs = rr['shape_attributes']
            if attrs['name'] == 'ellipse':
                 #I labelled some as ellipse by mistake, correct it
                 attrs['r'] = (attrs['rx'] + attrs['rx'])/2
            
            r, y, x = attrs['r'], attrs['cy'], attrs['cx']
            
            try:
                if rr['region_attributes']['Divided']['divided']:
                    lab = 2
                else:
                    raise ValueError
            except (ValueError, KeyError):
                lab = 1
                
            
            row = (img_id, x - r, y - r, x + r,  y + r,  r*2, lab)
            valid_data.append(row)
        
    
    df = pd.DataFrame(valid_data, columns=['img_id', 'x0','y0',  'x1', 'y1', 'bb_size', 'label'])
    return df


#def _get_augmented_crop(self, img, center):
#        im_c_padded = img[:, 
#                          center[1]-self.crop_right:center[1]+self.crop_right+1, 
#                          center[0]-self.crop_right:center[0]+self.crop_right+1
#                          ]
#        

#%%




        #%%
#        
#        cc = (self.crop_size_padded, self.crop_size_padded)
#        img_c_rot = [cv2.warpAffine(x, M, cc) for x in im_c_padded]
#        img_rot = np.array([x[self.pad_size:-self.pad_size, self.pad_size:-self.pad_size] for x in img_c_rot])
#%%        
class VesicleBBFlow(Dataset):
    def __init__(self, 
                 root_dir = root_dir,
                 crop_size = 256,
                 offset_overlap = 1/5,
                 files2sample = 5,
                 int_ranges = (7, 11.1),
                 zoom_ranges = (1.75, 2.25),
                 rot_ranges = (-180, 180),
                 file_bbox = file_bbox,
                 expand_factor = 10,
                 is_clean_data = False
                 ):
        
        if is_clean_data:
            img_dir = root_dir / 'training_data' / 'cleaned_frames'
        else:
            img_dir = root_dir / 'training_data' / 'initial_frames'
            
        self.zoom_ranges = zoom_ranges
        self.crop_size = crop_size
        self.int_ranges = int_ranges
        self.rot_ranges = rot_ranges
        self.img_dir = img_dir
        self.files2sample = files2sample
        self.expand_factor = expand_factor
        
        df = read_bb_attrs(file_bbox)
        df['offset'] = df['bb_size']*offset_overlap
        
        self.train_df = df
        self.index_per_img = df.groupby('img_id').groups
        
        
        
        #i need to see check the source directory to see if an image is train or test
        self._train_indexes = []
        self._test_indexes = []
        
        for img_id in self.index_per_img:
            bn = '{}/{}_0.png'.format(img_id, img_id)
            fname = self.img_dir / 'train' / bn
            if fname.exists():
                self._train_indexes.append(img_id)
            else:
                self._test_indexes.append(img_id)
            
        self.encoder = BoxEncoder((crop_size, crop_size))
        self.train()
        
        
#        self.crop_size = crop_size
#        self.pad_size = round((math.sqrt(2)-1)*self.crop_size/2)
#        self.crop_size_padded = 2*self.pad_size + self.crop_size
#        self.crop_left, self.crop_right = int(math.ceil(self.crop_size_padded/2)), int(math.floor(self.crop_size_padded/2))
#        self.center_rot = (self.crop_left, self.crop_right)
        
    def train(self):
        self.is_train = True
        self.valid_index = self._train_indexes
        self.valid_index = [x for _ in range(self.expand_factor) for x in self.valid_index]
        random.shuffle(self.valid_index)
    
    def test(self):
        self.is_train = False
        self.valid_index = self._test_indexes
        self.valid_index = [x for _ in range(self.expand_factor) for x in self.valid_index]
        random.shuffle(self.valid_index)
        
    def __len__(self):
        return len(self.valid_index)
    
    def __getitem__(self, ind):
        img_id = self.valid_index[ind]
        
        #randomly select an image from the directory (i am assuming the vesicle location does not change much from frame to frame)
        i_img = random.randint(0, self.files2sample)
        
        dd = 'train' if self.is_train else 'test'
        fname = self.img_dir / dd/ '{}/{}_{}.png'.format(img_id, img_id, i_img)
        
        #read image and normalize its intensity
        img_r = cv2.imread(str(fname), -1)
        
        img_r = np.log(img_r.astype(np.float32) + 1)
        
        img_r = (img_r-self.int_ranges[0])/(self.int_ranges[1]-self.int_ranges[0])
        
        for _ in range(20):
            #get the corresponding image rows
            vrows = self.train_df.loc[self.index_per_img[img_id]].copy()
            
            #random transforms
            img, vrows = self._random_zoom_rotation_crop(img_r, vrows)
            #img, vrows = self._random_zoom(img, vrows)
            img, vrows = self._random_vflip(img, vrows)
            img, vrows = self._random_hflip(img, vrows)
                
            if len(vrows)>0 and all(x==self.crop_size for x in img.shape):
                break
        
        #assert len(vrows>0 and all(x==self.crop_size for x in img.shape))
        
        bboxes = vrows[['x0', 'y0', 'x1', 'y1']].values
        labels = vrows['label'].values
        
        clf_target, loc_target = self.encoder.encode(labels, bboxes)
        
        img = img[None] #add channel dimenssion
        loc_target = loc_target.astype(np.float32)
        
        
        return img, clf_target, loc_target
        
        
    def _random_rotation_crop(self, img, vrows, _crop_size = None):
        if _crop_size is None:
            _crop_size = self.crop_size
        
        
        pad_size = round((math.sqrt(2)-1)*_crop_size/2)
        crop_size_padded = 2*pad_size + _crop_size
        crop_left, crop_right = int(math.ceil(crop_size_padded/2)), int(math.floor(crop_size_padded/2))
        center_rot = (crop_left, crop_right)
        
        
        img, vrows = self._random_crop(img, vrows, _crop_size = crop_size_padded)
        
        #pad in case the crop is not large enough to warranty the to rotation
        pad_ = [(crop_size_padded-x)/2 for x in img.shape]
        pad_ = [(int(math.floor(x)), int(math.ceil(x))) for x in pad_]
        img = np.pad(img, pad_, 'constant')
        vrows[['x0', 'x1']] += pad_[1][0]
        vrows[['y0', 'y1']] += pad_[0][0]
        
        rot_angle = random.randint(*self.rot_ranges)
        M = cv2.getRotationMatrix2D(center_rot, rot_angle, 1)
        
        cc = (crop_size_padded, crop_size_padded)
        img_c_rot = cv2.warpAffine(img, M, cc)
        img_r = img_c_rot[pad_size:-pad_size, pad_size:-pad_size]
        
        vrows_r = vrows.copy()
        R = vrows_r['bb_size']/2
        cm = (vrows_r[['x0','y0']].values + R[:,None])
        
        cm_r = np.matmul(np.pad(cm, ((0,0), (0,1)), 'constant', constant_values=1), M.T)
        
        vrows_r['x0'] = cm_r[:,0] - R
        vrows_r['y0'] = cm_r[:,1] - R
        vrows_r['x1'] = cm_r[:,0] + R
        vrows_r['y1'] = cm_r[:,1] + R
        
        vrows_r[['x0', 'y0', 'x1', 'y1']] -= pad_size
        
        
        return img_r, vrows_r
        
        
    def _random_zoom_rotation_crop(self, img, vrows):
        #random zoom
        eps = 1e-5
        smallest_size = min(img.shape)
        max_zoom = min(self.zoom_ranges[1], smallest_size/self.crop_size)
        min_zoom = min(self.zoom_ranges[0], self.zoom_ranges[1])
        
        img_scale = random.uniform(min_zoom-eps, max_zoom+eps)
        
        img, vrows = self._random_rotation_crop(img, vrows, _crop_size = int(round(self.crop_size*img_scale)))
        
        #img = cv2.resize(img, tuple(int(x/img_scale) for x in img.shape[::-1]))
        img = cv2.resize(img, (self.crop_size,self.crop_size))
        
        
        
        vrows[['x0', 'y0', 'x1', 'y1', 'bb_size', 'offset']] /= img_scale
        return img, vrows
        
    def _random_crop(self, img, vrows, _crop_size = None):
        if _crop_size is None:
            _crop_size = self.crop_size
        
        im_h, im_w = img.shape
    
        x1off = (vrows['x1']-vrows['offset'])
        x0off = (vrows['x0']+vrows['offset'])
        left_x, right_x = map(int, (x1off.min(), x0off.max()))
        
        left_x = max(0, min(left_x, im_w-_crop_size))
        right_x = max(left_x, right_x-_crop_size)
        
        x_anchor = random.randint(left_x, right_x)
        
        good = (x1off  > x_anchor) & (x0off < x_anchor+_crop_size)
        vrows = vrows[good]
        
        
        y1off = (vrows['y1']-vrows['offset'])
        y0off = (vrows['y0']+vrows['offset'])
        left_y, right_y = map(int, (y1off.min(), y0off.max()))
        
        left_y = max(0, min(left_y, im_h-_crop_size))
        right_y = max(left_y, right_y-_crop_size) 
        
        y_anchor = random.randint(left_y, right_y)
        
        
        good = (y1off > y_anchor) & (y0off < y_anchor+_crop_size)
        vrows = vrows[good]
        
        vrows_c = vrows.copy()
        vrows_c['x0'] -= x_anchor
        vrows_c['x1'] -= x_anchor
        vrows_c['y0'] -= y_anchor
        vrows_c['y1'] -= y_anchor
        
        img_c = img[y_anchor:y_anchor+_crop_size, x_anchor:x_anchor+_crop_size]
        
#        if any(x!=_crop_size for x in img_c.shape):
#            print('C')
#            import pdb
#            pdb.set_trace()
        
        return img_c, vrows_c
    
    def _random_vflip(self, img, vrows):
        if random.random() > 0.5:
            h = img.shape[0]
            img = img[::-1]
            
            vrows['y1'], vrows['y0'] = h - vrows['y0'], h - vrows['y1']
        
        return img, vrows
    
    def _random_hflip(self, img, vrows):
        if random.random() > 0.5:
            w = img.shape[1]
            
            img = img[:, ::-1]
            vrows['x1'], vrows['x0'] = w - vrows['x0'], w - vrows['x1']
            
        return img, vrows
    
     
def collate_fn(data):
    
    data = [x for x in data if x is not None]
    img, clf_target, loc_target = map(torch.from_numpy, map(np.stack, zip(*data)))

    return img, (clf_target, loc_target)

        