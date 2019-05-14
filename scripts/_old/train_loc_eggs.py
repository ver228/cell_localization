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

from cell_localization.trainer import train, get_device, log_dir_root_dflt
from cell_localization.flow import HistologyCoordFlow
from cell_localization.models import UNet, get_loss

def train_loc(
                data_type = 'eggs',
                model_name = 'unet',
                loss_type = 'l1smooth',
                cuda_id = 0,
                log_dir_root = log_dir_root_dflt,
                batch_size = 16,
                num_workers = 1,
                **argkws
                ):
    
    save_prefix = f'{data_type}_{model_name}_{loss_type}'
    log_dir = log_dir_root / 'eggs'
    
    n_ch_in, n_ch_out = 1,1
    
    batchnorm = '-bn' in model_name
    
    
    data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs/train'
    
    gen = HistologyCoordFlow(data_root_dir,
                             samples_per_epoch = 20480,
                             roi_size = 64,
                             zoom_range = (1., 1.),
                             loc_gauss_sigma = 1.5)
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out, batchnorm = batchnorm)
    device = get_device(cuda_id)
    
    criterion = get_loss(loss_type)
    
    
    train(save_prefix,
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
    
    fire.Fire(train_loc)
    