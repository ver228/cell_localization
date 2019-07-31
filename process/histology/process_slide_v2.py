#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:55:24 2019

@author: avelinojaver
"""
import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'scripts') )

from cell_localization.trainer import get_device
from cell_localization.models import CellDetector
from config_opts import model_types

import pandas as pd
import numpy as np
import tqdm
import shutil

from openslide import OpenSlide, lowlevel
import torch
import torch.nn.functional as F
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
    data_dir = Path().home() / 'workspace/localization/data/histology_bladder/TMA_counts/'
    
    is_switch_channels = True
    cuda_id = 0
    roi_size = 1024
    
    tmp_dir = None
    
    model_path = Path().home() / 'workspace/localization/results/locmax_detection/eosinophils/20x/eosinophils-20x/different_losses/eosinophils-20x+Feosinophils+roi96_unet-simple_l2-G2.5_20190728_054209_adam_lr0.000256_wd0.0_batch256/model_best.pth.tar'
    nms_threshold_abs = 0.15
    nms_threshold_rel = None
    loss_type = 'l2-G2.5'
    
#    model_path = Path().home() / 'workspace/localization/results/locmax_detection/eosinophils/20x/eosinophils-20x/different_models/eosinophils-20x+Feosinophilsonly+roi48+hard-neg-1_unet-simple_maxlikelihood_20190724_080046_adam_lr0.000256_wd0.0_batch256/model_best.pth.tar'
#    nms_threshold_abs = 0.0
#    nms_threshold_rel = 0.2
#    loss_type = 'maxlikelihood'
    
#    model_path = Path().home() / 'workspace/localization/results/locmax_detection/eosinophils/20x/eosinophils-20x/different_losses/eosinophils-20x+Feosinophils+roi48_unet-simple_l2-G2.5_20190727_020148_adam_lr0.000256_wd0.0_batch256/model_best.pth.tar'
#    nms_threshold_abs = 0.1
#    nms_threshold_rel = None
#    loss_type = 'l2-G2.5'
    
    n_ch_in = 3
    n_ch_out = 1
    
    
    model_args = model_types['unet-simple']
    model_args['unet_pad_mode'] = 'reflect'
    
    model = CellDetector(**model_args, 
                         unet_n_inputs = n_ch_in, 
                         unet_n_ouputs = n_ch_out,
                         loss_type = loss_type,
                         nms_threshold_abs = nms_threshold_abs,
                         nms_threshold_rel = nms_threshold_rel
                         )
    
    state = torch.load(model_path, map_location = 'cpu')
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    
    
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    
    assert model_path.exists()
    assert data_dir.exists()

    save_dir_root = Path.home() / 'workspace/localization/predictions/histology_detection' 
    #save_dir = Path.home() / 'workspace/localization/predictions/prostate-gland-phenotyping' 
    
    
    th_str = nms_threshold_abs if nms_threshold_rel is None else f'R{nms_threshold_rel}'
    save_dir = save_dir_root/ f'TH{th_str}_{model_path.parent.name}'
    save_dir.mkdir(exist_ok = True, parents = True)
    
    device = get_device(cuda_id)
    model = model.to(device)
    
    #%%
    if tmp_dir is not None and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    
    print(data_dir)
    fnames = [x for x in data_dir.rglob('*.svs') if not x.name.startswith('.')]
    fnames += [x for x in data_dir.rglob('*.mrxs') if not x.name.startswith('.')]
    
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
        
        batch_coords = []
        for xin, corners in tqdm.tqdm(loader):
            with torch.no_grad():
                if objective == '40':
                    xin = F.interpolate(xin, scale_factor = 0.5)
                assert (xin.shape[-2] == roi_size) & (xin.shape[-1] == roi_size)
                
                xin = xin.to(device)
                batch_predictions = model(xin)
            
            for predictions, corner in zip(batch_predictions, corners.numpy()):
                coords = predictions['coordinates'].detach().cpu().numpy()
                labels = predictions['labels'].detach().cpu().numpy()
                scores_abs = predictions['scores_abs'].detach().cpu().numpy()
                scores_rel = predictions['scores_rel'].detach().cpu().numpy()
                
                if objective == '40':
                    coords = coords*2
                coords = coords + corner[None]
                
                batch_coords.append((coords[:,0], coords[:,1], labels, scores_abs, scores_rel))
            
        cols = ['cx', 'cy', 'label', 'score_abs', 'scores_rel']
        dat = {c:np.concatenate(x) for c, x in zip(cols, zip(*batch_coords))}
        
        df = pd.DataFrame(dat)
        df.to_csv(save_name, index = False)
        
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir)