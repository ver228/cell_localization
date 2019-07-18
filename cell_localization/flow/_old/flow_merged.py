#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:27:27 2019

@author: avelinojaver
"""
import math
import random
from pathlib import Path
import numpy as np
import cv2
from torch.utils.data import Dataset 
#%%
class MergeFlow(Dataset):
    def __init__(self, 
                 ch1_dir = None,
                 ch2_dir = None,
                 img_ext = '.tif',
                 samples_per_epoch = 2000,
                 roi_size = 96,
                 zoom_range = (0.75, 1.5),
                 int_factor_range = (0.2, 1.2),
                 int_base_range = (0., 0.1),
                 patch_scale = True,
                 min_mix_frac = 0.3,
                 scale_int = (0, 255),
                 loc_sigma = 1.5,
                 is_scaled_output = False,
                 is_clipped_output = False,
                 is_preloaded = False,
                 add_inverse = False,
                 shuffle_ch2_color = False
                 ):
        
        self.ch1_dir = Path(ch1_dir)
        self.ch2_dir = Path(ch2_dir)
        
        self.samples_per_epoch = samples_per_epoch
        self.roi_size = roi_size
        self.roi_padding = math.ceil(roi_size*(math.sqrt(2)-1)/2)
        self.padded_roi_size = self.roi_size + 2*self.roi_padding
        
        self.zoom_range = zoom_range
        self.int_factor_range = int_factor_range
        self.int_base_range = int_base_range
        self.patch_scale = patch_scale
        
        self.scale_int = scale_int
        self.loc_sigma = loc_sigma
        
        assert min_mix_frac is None or min_mix_frac <= 0.5
        self.min_mix_frac = min_mix_frac
        
        
        self.is_scaled_output = is_scaled_output
        self.is_clipped_output = is_clipped_output
        self.add_inverse = add_inverse
        self.shuffle_ch2_color = shuffle_ch2_color
        
        self.is_preloaded = is_preloaded
        
        #print(self.ch1_dir)
        #print(self.ch2_dir)
        self.ch1_files = [x for x in self.ch1_dir.rglob('*' + img_ext) if not x.name.startswith('.')]
        self.ch2_files = [x for x in self.ch2_dir.rglob('*' + img_ext) if not x.name.startswith('.')]
        
        if self.is_preloaded:
            self.ch1_files = [cv2.imread(str(x), -1).astype(np.float32) for x in self.ch1_files]
            self.ch2_files = [cv2.imread(str(x), -1).astype(np.float32) for x in self.ch2_files]
        
        
    def __len__(self):
        return self.samples_per_epoch
            
    
    def __getitem__(self, ind):
        ch1_img = self._read_sample(self.ch1_files)
        ch2_img = self._read_sample(self.ch2_files)
        
        if self.shuffle_ch2_color:
            rand_factor = 0.3*np.random.random_sample(3) + 0.7
            rand_factor = rand_factor.astype(np.float32)
            ch_l = [0,1,2]
            random.shuffle(ch_l)
            ch2_img = ch2_img[ch_l, ...]*rand_factor[:, None, None]
        
        
        int_factor = random.uniform(*self.int_factor_range)
        int_base = random.uniform(*self.int_base_range)
        
        ch1_img *= int_factor
        ch1_img += int_base
        
        ch2_img *= int_factor
        ch2_img += int_base
        
        if self.min_mix_frac is not None:
            mix_factor = random.uniform(self.min_mix_frac, 1 - self.min_mix_frac)
            A, B = mix_factor*ch1_img, (1 - mix_factor)*ch2_img
        else:
            A, B = ch1_img, ch2_img
        
        if self.add_inverse:
            # 1 - ((1-A) + (1-B))
            Xin = A + B - 1 
            
        else:
            Xin = A + B
        
        
        if self.is_scaled_output:
            Xout = np.concatenate((ch1_img, ch2_img), axis=0)
        else:
            Xout = np.concatenate((A, B), axis=0)
            
        
        if self.is_clipped_output:
            Xout = np.clip(Xout, 0, 1)
            Xin = np.clip(Xin, 0, 1)
        
        
        
        return Xin, Xout
    
    def _read_sample(self, _files):
        if not self.is_preloaded:
            _file = random.choice(_files)
            img = cv2.imread(str(_file), -1)
            img = img.astype(np.float32)
        else:
            img = random.choice(_files)
        
        
        img = self._crop_augment(img)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.patch_scale:
            bot, top = img.min(), img.max()
            if bot < top:
                img_n = (img.astype(np.float32) - bot)/(top - bot)
            else:
                #image everything equal, nothing to do here...min_mix_frac
                img_n = np.zeros(img.shape, np.float32)
        else:
            img_n = (img.astype(np.float32) - self.scale_int[0])/(self.scale_int[1] - self.scale_int[0])
        
        if img_n.ndim == 2:
            img_n = img_n[None]
        else:
            img_n = np.rollaxis(img_n, 2, 0)
        
        return img_n
    
    
    def _crop_augment(self, img):
        #### select the limits allowed for a random crop
        xlims = (self.roi_padding, img.shape[1] - self.roi_size - self.roi_padding - 1)
        ylims = (self.roi_padding, img.shape[0] - self.roi_size - self.roi_padding - 1)
            
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
            
        
        ##### rotate
        theta = np.random.uniform(-180, 180)
        scaling = 1/np.random.uniform(*self.zoom_range)
        
        cols, rows = crop_padded.shape[0], crop_padded.shape[1]
        M = cv2.getRotationMatrix2D((rows/2,cols/2), theta, scaling)
        
        translation_matrix = np.array([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]])
        
        M = np.dot(M, translation_matrix)
        crop_rotated = cv2.warpAffine(crop_padded, M, (rows, cols), borderMode = cv2.BORDER_REFLECT_101)
        
        
        ##### remove padding
        crop_out = crop_rotated[self.roi_padding:-self.roi_padding, self.roi_padding:-self.roi_padding]
        
        ##### flips
        if random.random() > 0.5:
            crop_out = crop_out[::-1]
        
        if random.random() > 0.5:
            crop_out = crop_out[:, ::-1]
        
        
        
        return crop_out
    
    

#%%
if __name__ == '__main__':
    import matplotlib.pylab as plt
    
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/heba/WoundHealing/manually_filtered/'
    root_dir = Path(root_dir)
    flow_args = dict(
        ch1_dir = root_dir / 'nuclei',
        ch2_dir = root_dir / 'membrane',
        img_ext = '.tif',
        patch_scale = True
        )
    
    root_dir = Path.home() / 'workspace/denoising/data/inked_slides'
    flow_args = dict(
        ch1_dir = root_dir / 'clean',
        ch2_dir = root_dir / 'ink',
        img_ext = '.jpg',
        roi_size = 512,
        int_factor_range = (0.8, 1.2),
        int_base_range = (0., 0.05),
        min_mix_frac = None,
        patch_scale = False,
        is_scaled_output = True,
        is_clipped_output = True,
        shuffle_ch2_color = False,
        add_inverse = True
        )
                    
    gen = MergeFlow(**flow_args, is_preloaded = False)
    
#    for _ in range(1000):
#        X, target = gen[0]
#        assert not np.isnan(X).any()
#        assert not np.isnan(target).any()
#    
    for ii, (X,Y) in enumerate(gen):
        
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
        
        
        
        if Y.shape[0] == 6:
            y = np.rollaxis(Y, 0, 3)
            x = np.rollaxis(X, 0, 3)
            
            axs[0].imshow(x, vmin = 0., vmax = 1.)
            axs[1].imshow(y[..., :3], vmin = 0., vmax = 1.)
            axs[2].imshow(y[..., 3:], vmin = 0., vmax = 1.)
        else:
            axs[0].imshow(X[0], vmin = 0., vmax = 1.)
            axs[1].imshow(Y[0], vmin = 0., vmax = 1.)
            axs[2].imshow(Y[1], vmin = 0., vmax = 1.)
        
        axs[0].set_title('Ch1 + Ch2')
        axs[1].set_title('Ch1')
        axs[2].set_title('Ch2')
        if ii > 3:
            break
    
    