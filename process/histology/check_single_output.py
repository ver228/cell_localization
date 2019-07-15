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
from cell_localization.trainer import get_device

from process_slide import SlideFlow

import matplotlib.pylab as plt
import numpy as np
import tqdm
from openslide import OpenSlide, lowlevel


if __name__ == '__main__':
    data_dir = Path().home() / 'projects/bladder_cancer_tils/raw/'
    #data_dir = Path().home() / 'projects/bladder_cancer_tils/raw/40x/'
    
    is_switch_channels = True
    cuda_id = 0
    roi_size = 512
    
    #40x
    #model_path =  Path().home() / 'workspace/localization/results/histology_detection/bladder-cancer-tils-40x_unet_l1smooth_20190416_211621_adam_lr0.00064_wd0.0_batch64/model_best.pth.tar'
    #model_path =  Path().home() / 'workspace/localization/results/histology_detection/bladder-cancer-tils-40x_unet_l1smooth_20190416_211621_adam_lr0.00064_wd0.0_batch64/model_best.pth.tar'
    
    
    model_path = Path().home() / 'workspace/localization/results/locmax_detection/bladder/20x/bladder-tiles-roi64-20x/bladder-tiles-roi64-20x_unetv2_l1smooth_20190529_193223_adam_lr6.4e-05_wd0.0_batch64/model_best.pth.tar'
    
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    
    assert model_path.exists()
    assert data_dir.exists()

    save_dir = Path.home() / 'workspace/localization/predictions/histology_detection' / model_path.parent.name
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
    fnames = [x for x in data_dir.rglob('*.svs') if not x.name.startswith('.')]
    #fnames += [x for x in data_dir.rglob('*.mrxs') if not x.name.startswith('.')]
    
    fnames = [x for x in data_dir.rglob('*1048151Alast*') if not x.name.startswith('.')]
    #%%
    for slide_fname in tqdm.tqdm(fnames):
        assert slide_fname.exists()
        
        try:
            reader =  OpenSlide(str(slide_fname))
            objective = reader.properties['openslide.objective-power']
            
            reader.close()
        except lowlevel.OpenSlideUnsupportedFormatError:
            
            continue
#        
        if objective == '40':
            roi_size_l = roi_size*2
        else:
            roi_size_l = roi_size
           
        gen = SlideFlow(slide_fname, 
                        roi_size = roi_size_l,
                        slide_level = 0,
                        is_switch_channels = is_switch_channels
                        )
        
        roi, corner = gen[len(gen)//2]
    
#        slide_level = 0
#        reader =  OpenSlide(str(slide_fname))
#        _corner = (24212,7023)
#        _roi_size = (1308,405)
#        
#        roi = reader.read_region(_corner, slide_level, _roi_size)
#        roi = np.array(roi)[..., :-1]
#        
#        roi = np.rollaxis(roi, 2, 0)
#        if is_switch_channels:
#            roi = roi[::-1]
#        roi = roi.astype(np.float32)/255
        
        xin = torch.from_numpy(roi)[None]
        with torch.no_grad():
            xin = xin.to(device)
            if objective == '40':
               xin = F.interpolate(xin, scale_factor = 0.5)
            
            xhat = model(xin)
        xhat = xhat.detach().cpu().numpy()
        
        #%%
        img = np.rollaxis(roi, 0, 3)
        if is_switch_channels:
            img = img[..., ::-1]
        
        fig, axs = plt.subplots(1, 3, sharex = True, sharey = True)
        axs[0].imshow(img)
        axs[1].imshow(xhat[0, 0], vmin=0, vmax=1)
        axs[2].imshow(xhat[0, 1], vmin=0, vmax=1)
        