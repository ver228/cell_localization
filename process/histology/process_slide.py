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
from openslide import OpenSlide, lowlevel

from torch.utils.data import Dataset, DataLoader

class SlideFlow(Dataset):
    def __init__(self, 
                 fname,
                 roi_size = 1024,
                 roi_pad = 64
                 ):
        
        self.fname = fname
        self.roi_size = (roi_size, roi_size)
        self.roi_pad = roi_pad
        
        reader =  OpenSlide(str(self.fname))
        slide_size = reader.dimensions
        reader.close()
        
        rr = roi_size - roi_pad
        self.corners = [(x,y) for x in range(0, slide_size[0], rr) for y in range(0, slide_size[1], rr)]
    
    
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
        except lowlevel.OpenSlideUnsupportedFormatError:
            continue
            
        gen = SlideFlow(slide_fname)
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
        
        