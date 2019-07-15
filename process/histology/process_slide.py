#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:55:24 2019

@author: avelinojaver
"""


import sys
from pathlib import Path 
dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from pathlib import Path
import torch
import torch.nn.functional as F

from cell_localization.models import UNet, UNetv2
from cell_localization.evaluation.localmaxima import cv2_peak_local_max
from cell_localization.trainer import get_device

import pandas as pd
import numpy as np
import tqdm
import shutil

from openslide import OpenSlide, lowlevel

from torch.utils.data import Dataset, DataLoader

class SlideFlow(Dataset):
    def __init__(self, 
                 fname,
                 roi_size = 1024,
                 roi_pad = 32,
                 slide_level = 0,
                 is_switch_channels = False
                 ):
        print(fname)
        
        self.fname = fname
        self.roi_size = (roi_size, roi_size)
        self.roi_pad = roi_pad
        self.is_switch_channels = is_switch_channels
        
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
        if self.is_switch_channels:
            roi = roi[..., ::-1]
        
        roi = np.rollaxis(roi, 2, 0)
        roi = roi.astype(np.float32)/255
        
        return roi, np.array(corner)



if __name__ == '__main__':
    data_dir = Path().home() / 'projects/bladder_cancer_tils/raw/'
    #data_dir = Path('/well/rittscher/projects/prostate-gland-phenotyping/WSI')
    
    is_switch_channels = True
    cuda_id = 0
    roi_size = 1024
    
    #tmp_dir = Path('/tmp/avelino/slide')
    tmp_dir = None
    
    #40x
    #model_path =  Path().home() / 'workspace/localization/results/histology_detection/bladder-cancer-tils-40x_unet_l1smooth_20190416_211621_adam_lr0.00064_wd0.0_batch64/model_best.pth.tar'
    #model_path =  Path().home() / 'workspace/localization/results/histology_detection/bladder-cancer-tils-40x_unet_l1smooth_20190416_211621_adam_lr0.00064_wd0.0_batch64/model_best.pth.tar'
    model_path = Path().home() / 'workspace/localization/results/locmax_detection/bladder/20x/bladder-tiles-roi64-20x/bladder-tiles-roi64-20x_unetv2_l1smooth_20190529_193223_adam_lr6.4e-05_wd0.0_batch64/model_best.pth.tar'
    threshold_abs = 0.1
    
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    
    assert model_path.exists()
    assert data_dir.exists()

    save_dir_root = Path.home() / 'workspace/localization/predictions/histology_detection' 
    #save_dir = Path.home() / 'workspace/localization/predictions/prostate-gland-phenotyping' 
    
    save_dir = save_dir_root/ f'TH{threshold_abs}_{model_path.parent.name}'
    
    save_dir.mkdir(exist_ok = True, parents = True)


    
    n_ch_in, n_ch_out  = 3, 2
    bn = model_path.parent.name
    batchnorm = 'unet-bn' in bn
    if 'unetv2' in bn:
        model = UNetv2(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm=batchnorm)
    else:
        model = UNet(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm=batchnorm)
    
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    device = get_device(cuda_id)
    model = model.to(device)
    
    #%%
    
    
    if tmp_dir is not None and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    
    print(data_dir)
    fnames = [x for x in data_dir.rglob('*.svs') if not x.name.startswith('.')]
    fnames += [x for x in data_dir.rglob('*.mrxs') if not x.name.startswith('.')]
    #fnames = [x for x in data_dir.rglob('*1048151Alast*') if not x.name.startswith('.')]
    
#    ss = ['17_A047-4463_153D+-+2017-05-11+09.40.22.ndpi',
#            '17_A047-10719_16L+-+2017-05-11+08.56.52.ndpi',
#            'EU_29542_16_1S_HandE+-+2017-11-28+12.49.46.ndpi']
#    fnames = [data_dir / x for x in ss]
    
    for slide_fname_src in tqdm.tqdm(fnames[::-1]):
        assert slide_fname_src.exists()
        
        try:
            reader =  OpenSlide(str(slide_fname_src))
            objective = reader.properties['openslide.objective-power']
            reader.close()
        except (lowlevel.OpenSlideUnsupportedFormatError, lowlevel.OpenSlideError):
            print('BAD', slide_fname_src)
            continue
        
        if  tmp_dir is None:
            slide_fname = slide_fname_src
        else:
            tmp_dir.mkdir(exist_ok = True, parents = True)
            shutil.copy(slide_fname, tmp_dir)
            if slide_fname.suffix == '.mrxs':
                shutil.copytree(slide_fname.parent /  slide_fname.stem, tmp_dir  /  slide_fname.stem)
        
            tmp_file = tmp_dir / slide_fname.name
            assert tmp_file.exists()
            
            slide_fname = tmp_file
            
        
        save_name = save_dir / slide_fname.parent.name / (slide_fname.stem + '.csv')
        if save_name.exists():
            continue
        
        save_name.parent.mkdir(exist_ok=True, parents=True)
        
        if objective == '40':
            roi_size_l = roi_size*2
        else:
            roi_size_l = roi_size
        
        gen = SlideFlow(slide_fname, 
                        roi_size = roi_size_l,
                        slide_level = 0,
                        is_switch_channels = is_switch_channels
                        )
        loader = DataLoader(gen, 
                            batch_size = 4, 
                            shuffle=True, 
                            num_workers = 6)
        
        coords = []
        for xin, corners in tqdm.tqdm(loader):
            with torch.no_grad():
                if objective == '40':
                    xin = F.interpolate(xin, scale_factor = 0.5)
                assert (xin.shape[-2] == roi_size) & (xin.shape[-1] == roi_size)
                
                xin = xin.to(device)
                xhat = model(xin)
            xhat = xhat.detach().cpu().numpy()
            
            for xout, corner in zip(xhat, corners.numpy()):
                for mm, t_lab in zip(xout, ['L','E']):
                    map_coords = cv2_peak_local_max(mm, threshold_relative = 0.0, threshold_abs = threshold_abs)
                    
                    if map_coords.size > 0:
                        if objective == '40':
                            map_coords *= 2
                        
                        map_coords += corner[None]
                        coords += [(t_lab, *c) for c in map_coords]
        
        df = pd.DataFrame(coords)
        df.to_csv(save_name, index = False)
        
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir)