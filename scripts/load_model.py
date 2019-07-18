#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:29:14 2019

@author: avelinojaver
"""
#%%
from config_opts import flow_types, data_types, model_types

from cell_localization.models import CellDetector
from cell_localization.flow import CoordFlow

import torch

def load_model(model_dir, bn, flow_subdir = 'validation'):
    data_type, _, remain = bn.partition('+F')
    flow_type, _, remain = remain.partition('+roi')
    
    
    model_name, _, remain = remain.partition('unet-')[-1].partition('_')
    model_name = 'unet-' + model_name
    
    remain = remain.split('_')
    loss_type = remain[0]
    
    
    model_path = model_dir / data_type / bn / 'checkpoint.pth.tar'
    state = torch.load(model_path, map_location = 'cpu')
    
    data_args = data_types[data_type]
    n_ch_in = data_args['n_ch_in']
    n_ch_out = data_args['n_ch_out']
    
    model_args = model_types[model_name]
    model_args['unet_pad_mode'] = 'reflect'
    
    model = CellDetector(**model_args, 
                         unet_n_inputs = n_ch_in, 
                         unet_n_ouputs = n_ch_out,
                         loss_type = loss_type,
                         return_belive_maps = True
                         )
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    flow_args = flow_types[flow_type]
    data_flow = CoordFlow(data_args['root_data_dir'] / flow_subdir,
                        **flow_args,
                        is_preloaded = False
                        ) 
    
    return model, data_flow