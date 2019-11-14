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
from cell_localization.trainer import train_mask
from cell_localization.flow import MasksFlow
from cell_localization.models import get_mapping_network, model_types

import datetime
import torch
from torch import nn

def get_criterion(loss_name):
    if loss_name == 'BCE':
        return nn.BCELoss()
    else:
        raise ValueError(f'Not implemented `{loss_name}`.')

LOG_DIR_DFLT = Path.home() / 'workspace/segmentation/results/'




def train(
        data_type = 'woundhealing-contour',
        flow_type = None,
        model_name = 'unet-flat-48',
        loss_type = 'BCE',
        cuda_id = 0,
        log_dir = None,
        batch_size = 56,
        n_epochs = 2000,
        save_frequency = 200,
        num_workers = 0,
        root_data_dir = None,
        
        optimizer_name = 'adam',
        lr_scheduler_name = '',
        lr = 64e-5,
        weight_decay = 0.0,
        momentum = 0.9,
        
        roi_size = 256,
        
        
        model_path_init = None,
        train_samples_per_epoch = 16384,
        
        num_folds = 5,
        val_fold_id = 1,
        
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
    
    
    data_type += f'fold-{val_fold_id}-{num_folds}'
    
    train_fold_ids = [x + 1 for x in range(num_folds) if x + 1 != val_fold_id]
    train_flow = MasksFlow(root_data_dir,
                    samples_per_epoch = train_samples_per_epoch,
                    roi_size = roi_size,
                    **flow_args,
                    folds2include = train_fold_ids,
                    num_folds = num_folds
                    )  
    
    val_flow = MasksFlow(root_data_dir,
                    roi_size = roi_size,
                    **flow_args,
                    folds2include = val_fold_id,
                    num_folds = num_folds
                    ) 
        
        
    
    model = get_mapping_network(n_ch_in, n_ch_out, **model_types[model_name], output_activation = 'sigmoid')
    criterion = get_criterion(loss_type)
    
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
    lr_scheduler_name = '+' + lr_scheduler_name if lr_scheduler_name else ''
    
    
    save_prefix = f'{data_type}+F{flow_type}+roi{roi_size}_{model_name}_{loss_type}_{date_str}'
    save_prefix = f'{save_prefix}_{optimizer_name}{lr_scheduler_name}_lr{lr}_wd{weight_decay}_batch{batch_size}'
    
    
    train_mask(save_prefix,
        model,
        device,
        criterion,
        train_flow,
        val_flow,
        optimizer,
        log_dir,
        
        lr_scheduler,
        
        batch_size,
        n_epochs,
        num_workers
        )

        
if __name__ == '__main__':
    import fire
    fire.Fire(train)
    