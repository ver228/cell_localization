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


import datetime

from cell_localization.trainer import train_locmax, get_device, log_dir_root_dflt, get_optimizer
from cell_localization.flow import CoordFlow
from cell_localization.models import UNet, get_loss

data_types_dflts = {
    'heba': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 96,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-int': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 96,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.75, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'eggs': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs',
        flow_args = dict(
                roi_size = 64,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 1.5
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'eggs-int': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs',
        flow_args = dict(
                roi_size = 64,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 1.5,
                int_aug_offset = (-0.15, 0.15),
                int_aug_expansion = (0.85, 1.2)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'eggs-only': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs',
        flow_args = dict(
                roi_size = 64,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.0,
                loc_gauss_sigma = 1.5,
                int_aug_offset = (-0.15, 0.15),
                int_aug_expansion = (0.85, 1.2)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'eggs-int-old': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/old_worm_eggs',
        flow_args = dict(
                roi_size = 64,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 1.5,
                int_aug_offset = (-0.15, 0.15),
                int_aug_expansion = (0.85, 1.2)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'bladder-20x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/20x',
        roi_size = 64,
        scale_int = (0, 255),
        prob_unseeded_patch = 0.0,
        loc_gauss_sigma = 2.5,
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
    
    'bladder-40x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/40x',
        roi_size = 128,
        scale_int = (0, 255),
        prob_unseeded_patch = 0.0,
        loc_gauss_sigma = 5,
        n_ch_in = 3,
        n_ch_out = 2
        
        ) 
        
    }

def train(
        data_type = 'heba',
        model_name = 'unet',
        loss_type = 'l1smooth',
        cuda_id = 0,
        log_dir = None,
        batch_size = 16,
        num_workers = 1,
        root_data_dir = None,
        optimizer_name = 'adam',
        lr = 1e-5,
        weight_decay = 0.0,
        momentum = 0.9,
        is_preloaded = False,
        hard_mining_freq = None,
        **argkws
        ):
    
    if log_dir is None:
        log_dir = log_dir_root_dflt / 'locmax_detection' / data_type
    
    
    dflts = data_types_dflts[data_type]
    dflt_root_data_dir = dflts['root_data_dir']
    flow_args = dflts['flow_args']
    n_ch_in = dflts['n_ch_in']
    n_ch_out = dflts['n_ch_out']
    
    if root_data_dir is None:
        root_data_dir = dflt_root_data_dir
    root_data_dir = Path(root_data_dir)
    
    train_dir = root_data_dir / 'train'
    test_dir = root_data_dir / 'validation'
    
    
    print(root_data_dir)
    train_flow = CoordFlow(train_dir,
                    samples_per_epoch = 40960,
                    **flow_args,
                    is_preloaded = is_preloaded
                    )  
    
    val_flow = CoordFlow(test_dir,
                    samples_per_epoch = 640,
                    **flow_args,
                    is_preloaded = is_preloaded
                    ) 
    
    batchnorm = '-bn' in model_name
    out_sigmoid = '-sigmoid' in model_name
    
    model = UNet(n_channels = n_ch_in, 
                 n_classes = n_ch_out, 
                 batchnorm = batchnorm,
                 out_sigmoid = out_sigmoid
                 )
    device = get_device(cuda_id)
    
    
    device = get_device(cuda_id)
    
    criterion = get_loss(loss_type)
    
    optimizer = get_optimizer(optimizer_name, 
                              model, 
                              lr = lr, 
                              momentum = momentum, 
                              weight_decay = weight_decay)
    
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    
    hard_mining_str = '' if hard_mining_freq is None else f'_hard-neg-freq{hard_mining_freq}'
    save_prefix = f'{data_type}_{model_name}{hard_mining_str}_{loss_type}_{date_str}'
    save_prefix = f'{save_prefix}_{optimizer_name}_lr{lr}_batch{batch_size}'
    
    train_locmax(save_prefix,
        model,
        device,
        train_flow,
        val_flow,
        criterion,
        optimizer,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        hard_mining_freq = hard_mining_freq,
        **argkws
        )
    

if __name__ == '__main__':
    import fire
    
    fire.Fire(train)
    