#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:55:24 2019

@author: avelinojaver
"""


import sys
from pathlib import Path 
dname = Path(__file__).resolve().parents[1]
sys.path.append(str(dname))


from pathlib import Path
import torch
from cell_localization.models import UNet
from cell_localization.evaluation.localmaxima import cv2_peak_local_max
from cell_localization.trainer import get_device

import cv2
import pandas as pd
import numpy as np
import tqdm
from openslide import OpenSlide

from torch.utils.data import Dataset, DataLoader

def get_single_candidate(img_rgb, low_th = 70, high_th = 230, std_th=5):
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))    
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))    
    
    rr = np.std(img_rgb, axis=2)
    _, valid_mask_p = cv2.threshold(rr, std_th, 255, cv2.THRESH_BINARY)
    valid_mask_p = valid_mask_p.astype(np.uint8)
    
    
    dark_mask_p = np.any(img_rgb<low_th, axis=-1).astype(np.uint8)*255
    bright_mask_p = np.any(img_rgb>high_th, axis=-1).astype(np.uint8)*255
    
    
    dark_mask = cv2.erode(dark_mask_p, k1)
    dark_mask = cv2.dilate(dark_mask, k2)
    
    bright_mask = cv2.erode(bright_mask_p, k1)
    bright_mask = cv2.dilate(bright_mask, k2)
    
    valid_mask = cv2.bitwise_not(cv2.bitwise_or(dark_mask, bright_mask))
    valid_mask = cv2.bitwise_and(valid_mask, valid_mask_p)
    
    im2, cnts, hierarchy = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_cnt = max(cnts, key = cv2.contourArea)
    
    best_mask = np.zeros_like(valid_mask)
    cv2.drawContours(best_mask, [largest_cnt], 0, 255, -1)
    
    #valid_mask_p = cv2.bitwise_not(bright_mask)
    best_mask = cv2.bitwise_and(best_mask, valid_mask_p)
    return best_mask

class SlideFlowMask(Dataset):
    def __init__(self, 
                 fname,
                 roi_size = 1024,
                 roi_pad = 64
                 ):
        
        self.fname = fname
        self.roi_size = (roi_size, roi_size)
        self.roi_pad = roi_pad
        
    
        reader =  OpenSlide(str(self.fname))
        
        level_n = min(reader.level_count - 1, 7)
        level_dims = reader.level_dimensions[level_n]
        downsample = reader.level_downsamples[level_n]
        corner = (0,0)
        
        
        img = reader.read_region(corner, level_n, level_dims)
        
        img = np.array(img)
        img_rgb = img[..., :-1]
        
        
        best_mask = get_single_candidate(img_rgb, low_th = 70, high_th = 230)
        
        _factor = downsample/roi_size
        tiny_mask = cv2.resize(best_mask, dsize=(0,0), fx=_factor, fy=_factor)
        tiny_mask = tiny_mask==255
        
        
        v_rows, v_cols = np.where(tiny_mask)
        corners = list(zip(v_cols*roi_size, v_rows*roi_size))
        
        reader.close()
        
        self.corners = corners
    
    
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
        try:
            roi = reader.read_region(corner, 0, self.roi_size)
        except:
            
            raise(self.fname)
        reader.close()
        
        roi = np.array(roi)[..., :-1]
        
        roi = np.rollaxis(roi, 2, 0)
        roi = roi.astype(np.float32)/255
        
        return roi, np.array(corner)



if __name__ == '__main__':
    cuda_id = 0
    
    device = get_device(cuda_id)
    
    
    #20x
    #model_path =  Path().home() / 'workspace/localization/results/histology_detection/bladder-cancer-tils_unet_l1smooth_20190406_000552_adam_lr0.00064_wd0.0_batch64/model_best.pth.tar'
    #data_dir =  Path().home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/raw/20x/'
    #data_dir =  Path().home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/raw/new_20x/'
    
    #data_dir = '/tmp/avelino/new_20x/'
    #data_dir = '/tmp/avelino/20x/'
    
    
    #40x
    model_path =  Path().home() / 'workspace/localization/results/histology_detection/bladder-cancer-tils-40x_unet_l1smooth_20190416_211621_adam_lr0.00064_wd0.0_batch64/model_best.pth.tar'
    #data_dir = '/tmp/avelino/new_40x/'
    #data_dir = '/tmp/avelino/40x/'
    data_dir = Path().home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/raw/HEs'
    
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    
    assert model_path.exists()
    assert data_dir.exists()

    save_dir = Path.home() / 'workspace/localization/predictions/histology_detection' / model_path.parent.name
    save_dir.mkdir(exist_ok = True, parents = True)

    n_ch_in, n_ch_out  = 3, 2
    batchnorm = 'unet-bn' in model_path.parent.name
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm=batchnorm)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    model = model.to(device)
    
    #%%
    
    fnames = [x for x in data_dir.glob('*.svs') if not x.name.startswith('.')]
    fnames += [x for x in data_dir.rglob('*.mrxs') if not x.name.startswith('.')]
    for slide_fname in tqdm.tqdm(fnames):
        assert slide_fname.exists()
        save_name = save_dir / slide_fname.parent.name / (slide_fname.stem + '.csv')
        if save_name.exists():
            continue
        
        save_name.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            reader =  OpenSlide(str(slide_fname))
            reader.close()
        except:
            continue
        
        print(slide_fname)
        gen = SlideFlowMask(slide_fname)
        loader = DataLoader(gen, 
                            batch_size = 4, 
                            shuffle=True, 
                            num_workers = 12)
        
        coords = []
        for xin, corners in tqdm.tqdm(loader):
            with torch.no_grad():
                #xin = torch.from_numpy(xin[None])
                xin = xin.to(device)
                xhat = model(xin)
            xhat = xhat.detach().cpu().numpy()
            
            for xout, corner in zip(xhat, corners.numpy()):
                for mm, t_lab in zip(xout, ['L','E']):
                    map_coords = cv2_peak_local_max(mm, threshold_relative = 0.1, threshold_abs = 0.5)
                    
                    if map_coords.size > 0:
                        map_coords += corner[None]
                        coords += [(t_lab, *c) for c in map_coords]
        
        df = pd.DataFrame(coords)
        df.to_csv(save_name, index = False)
        
        