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

from config_opts import flow_types, data_types

from cell_localization.utils import get_device, get_scheduler, get_optimizer
from cell_localization.trainer import train_locmax
from cell_localization.flow import CoordFlow
from cell_localization.models import get_model

import datetime
import torch

LOG_DIR_DFLT = Path.home() / 'workspace/localization/results/locmax_detection'
def train(
        data_type = 'woundhealing-v2-mix',
        flow_type = None,
        model_name = 'unet-simple',
        loss_type = 'l1smooth-G1.5',
        cuda_id = 0,
        log_dir = None,
        batch_size = 256,
        n_epochs = 2000,
        save_frequency = 200,
        num_workers = 0,
        root_data_dir = None,
        
        optimizer_name = 'adam',
        lr_scheduler_name = '',
        lr = 1e-5,
        weight_decay = 0.0,
        momentum = 0.9,
        
        roi_size = 64,
        
        is_preloaded = False,
        
        hard_mining_freq = None,
        model_path_init = None,
        train_samples_per_epoch = 40960,
        
        num_folds = None,
        val_fold_id = None,
        
        val_dist = 5
        ):
    
    
    data_args = data_types[data_type]
    dflt_root_data_dir = data_args['root_data_dir']
    
    n_ch_in = data_args['n_ch_in']
    n_ch_out = data_args['n_ch_out']
    
    if flow_type is None:
        flow_type = data_args['dflt_flow_type']
    
    flow_args = flow_types[flow_type]
    
    
    if log_dir is None:
        if 'log_prefix' in data_args:
            log_dir = LOG_DIR_DFLT / data_args['log_prefix'] / data_type
        else:
            log_dir = LOG_DIR_DFLT / data_type
    
    if root_data_dir is None:
        root_data_dir = dflt_root_data_dir
    root_data_dir = Path(root_data_dir)
    
    if root_data_dir.is_dir():
        assert val_fold_id is None
        
        train_dir = root_data_dir / 'train'
        test_dir = root_data_dir / 'validation'
        
        
        
        print(root_data_dir)
        train_flow = CoordFlow(train_dir,
                        samples_per_epoch = train_samples_per_epoch,
                        roi_size = roi_size,
                        **flow_args,
                        is_preloaded = is_preloaded
                        )  
        
        val_flow = CoordFlow(test_dir,
                        roi_size = roi_size,
                        **flow_args,
                        is_preloaded = is_preloaded
                        ) 
    else:
        data_type += f'fold-{val_fold_id}-{num_folds}'
        
        train_fold_ids = [x + 1 for x in range(num_folds) if x + 1 != val_fold_id]
        train_flow = CoordFlow(root_data_dir,
                        samples_per_epoch = train_samples_per_epoch,
                        roi_size = roi_size,
                        **flow_args,
                        folds2include = train_fold_ids,
                        num_folds = num_folds,
                        is_preloaded = True
                        )  
        
        val_flow = CoordFlow(root_data_dir,
                        roi_size = roi_size,
                        **flow_args,
                        folds2include = val_fold_id,
                        num_folds = num_folds,
                        is_preloaded = True
                        ) 
        
        
    
    model = get_model(model_name, n_ch_in, n_ch_out, loss_type)
    
    if model_path_init is not None:
        model_name += '-pretrained'
        state = torch.load(model_path_init, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
    
    device = get_device(cuda_id)
    
    optimizer = get_optimizer(optimizer_name, 
                              model, 
                              lr = lr, 
                              momentum = momentum, 
                              weight_decay = weight_decay)
    
    lr_scheduler = get_scheduler(lr_scheduler_name, optimizer)
    
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hard_mining_str = '' if hard_mining_freq is None else f'+hard-neg-{hard_mining_freq}'
    lr_scheduler_name = '+' + lr_scheduler_name if lr_scheduler_name else ''
    
    
    save_prefix = f'{data_type}+F{flow_type}+roi{roi_size}{hard_mining_str}_{model_name}_{loss_type}_{date_str}'
    save_prefix = f'{save_prefix}_{optimizer_name}{lr_scheduler_name}_lr{lr}_wd{weight_decay}_batch{batch_size}'
    
    train_locmax(save_prefix,
        model,
        device,
        train_flow,
        val_flow,
        optimizer,
        lr_scheduler = lr_scheduler,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        hard_mining_freq = hard_mining_freq,
        n_epochs = n_epochs,
        save_frequency = save_frequency,
        val_dist = val_dist
        )

if __name__ == '__main__':
    import fire
    fire.Fire(train)
    