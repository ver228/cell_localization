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
                data_type = 'bladder-cancer-tils-40x',
                model_name = 'unet-bn',
                loss_type = 'l1smooth',
                cuda_id = 0,
                log_dir_root = log_dir_root_dflt,
                batch_size = 16,
                num_workers = 1,
                roi_size = 128,
                data_root_dir = None,
                **argkws
                ):
    
    save_prefix = f'{data_type}_{model_name}_{loss_type}'
    log_dir = log_dir_root / 'histology_detection'
    
    n_ch_in, n_ch_out = 3, 2
    
    
    if data_root_dir is None:
        _root_dir = Path.home() / 'workspace/localization/data/histology_data/bladder_cancer_tils/rois/'
        if '20x' in data_type:
            data_root_dir = _root_dir / '20x' / 'train'
        elif '40x' in data_type:
            data_root_dir = _root_dir / '40x' / 'train'
        else:
            raise ValueError(f'Not implemented {data_type}')
    else:
        data_root_dir = Path(data_root_dir)
        assert data_root_dir.exists()
    
    gen = HistologyCoordFlow(data_root_dir,
                    samples_per_epoch = 20480,
                    roi_size = roi_size
                    )  
    batchnorm = '-bn' in model_name
    
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
    