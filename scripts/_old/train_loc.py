#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from cell_localization.trainer import train_locmax, get_device, log_dir_root_dflt
from cell_localization.flow import CoordFlow
from cell_localization.models import UNet, get_loss

def train(
        data_type = 'cell-loc',
        model_name = 'unet',
        loss_type = 'l1smooth',
        cuda_id = 0,
        log_dir = None,
        batch_size = 16,
        num_workers = 1,
        data_dir = None,
        **argkws
        ):
    
    if log_dir is None:
        log_dir = log_dir_root_dflt / 'cell_detection'
    
    
    if data_dir is None:
        data_dir = Path.home() / 'workspace/localization/cell_detection/data/train'
    
    save_prefix = f'{data_type}_{model_name}_{loss_type}'
    
    print(data_dir)
    gen = CoordFlow(data_dir,
                    samples_per_epoch = 20480,
                    loc_gauss_sigma = 2,
                    roi_size = 96,
                    scale_int = (0, 4095)
                    )  

    n_ch_in, n_ch_out = 1, 1
    if model_name == 'unet-bn':
        model = UNet(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm = True)
    elif model_name == 'unet':
        model = UNet(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm = False)

    device = get_device(cuda_id)
    criterion = get_loss(loss_type)
    
    train_locmax(save_prefix,
        model,
        device,
        gen,
        criterion,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        **argkws
        )
    

if __name__ == '__main__':
    import fire
    
    fire.Fire(train)
    