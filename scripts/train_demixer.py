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

from cell_localization.trainer import train_demixer, get_device, log_dir_root_dflt, get_optimizer
from cell_localization.flow import MergeFlow
from cell_localization.models import UNet, get_loss

import datetime

ink_root_dir = root_dir = Path.home() / 'workspace/denoising/data/inked_slides'
woundhealing_root_dir = Path.home() / 'workspace/localization/data/woundhealing/manually_filtered'

data_types_dflts = {
    'woundhealing': dict(
    flow_args = dict(
        ch1_dir = woundhealing_root_dir / 'nuclei',
        ch2_dir = woundhealing_root_dir / 'membrane',
        img_ext = '.tif',
        patch_scale = True
        ),
    n_ch_in = 1,
    n_ch_out = 2
    ),
    'inkedslides': dict(
    flow_args = dict(
        ch1_dir = ink_root_dir / 'clean',
        ch2_dir = ink_root_dir / 'ink',
        img_ext = '.jpg',
        int_factor_range = (0.8, 1.2),
        int_base_range = (0., 0.05),
        min_mix_frac = None,
        patch_scale = False,
        is_scaled_output = True,
        shuffle_ch2_color = True,
        add_inverse = True
        ),
    n_ch_in = 3,
    n_ch_out = 6
    ),
    'inkedslidesv2': dict(
    flow_args = dict(
        ch1_dir = ink_root_dir / 'clean',
        ch2_dir = ink_root_dir / 'ink',
        img_ext = '.jpg',
        int_factor_range = (0.9, 1.1),
        int_base_range = (0., 0.05),
        min_mix_frac = None,
        patch_scale = False,
        is_scaled_output = True,
        shuffle_ch2_color = False,
        add_inverse = True
        ),
    n_ch_in = 3,
    n_ch_out = 6
    ),
    'inkedslidesv3': dict(
    flow_args = dict(
        ch1_dir = ink_root_dir / 'clean',
        ch2_dir = ink_root_dir / 'ink',
        img_ext = '.jpg',
        int_factor_range = (0.9, 1.1),
        int_base_range = (0., 0.05),
        min_mix_frac = None,
        patch_scale = False,
        is_scaled_output = True,
        is_clipped_output = True,
        shuffle_ch2_color = False,
        add_inverse = True
        ),
    n_ch_in = 3,
    n_ch_out = 6
    )
    }

def train(
        data_type = 'woundhealing',
        model_name = 'unet',
        loss_type = 'l1smooth',
        cuda_id = 0,
        log_dir_root = log_dir_root_dflt,
        batch_size = 16,
        optimizer_name = 'adam',
        lr = 1e-5,
        weight_decay = 0.0,
        momentum = 0.9,
        roi_size = 128,
        is_scaled_output = False,
        is_preloaded = False,
        **argkws
        ):

    
    
    log_dir = log_dir_root / 'cell_demixer' / data_type
    
    
    dflts = data_types_dflts[data_type]
    n_ch_in = dflts['n_ch_in']
    n_ch_out = dflts['n_ch_out']
    flow_args = dflts['flow_args']
    
    train_flow = MergeFlow(**flow_args,
                    roi_size = roi_size, 
                    is_preloaded = is_preloaded,
                    samples_per_epoch = 20480
                    )  
    
    model = UNet(n_channels = n_ch_in, n_classes = n_ch_out)
    device = get_device(cuda_id)
    
    criterion = get_loss(loss_type)
    
    optimizer = get_optimizer(optimizer_name, 
                              model, 
                              lr = lr, 
                              momentum = momentum, 
                              weight_decay = weight_decay)
    
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if is_scaled_output:
        data_type += '-scaled'
    save_prefix = f'{data_type}-roi{roi_size}_{model_name}_{loss_type}_{date_str}'
    save_prefix = f'{save_prefix}_{optimizer_name}_lr{lr}_wd{weight_decay}_batch{batch_size}'
    
    train_demixer(save_prefix,
        model,
        device,
        train_flow,
        criterion,
        optimizer,
        log_dir,
        batch_size = batch_size,
        **argkws
        )
    
    
    

if __name__ == '__main__':
    import fire
    
    fire.Fire(train)
    