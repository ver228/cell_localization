#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:29:14 2019

@author: avelinojaver
"""
#%%
from config_opts import flow_types, data_types, model_types

from cell_localization.models import CellDetector
from cell_localization.flow import CoordFlow, CoordFlowMerged

import torch
from pathlib import Path

def load_model(model_path, flow_subdir = 'validation', data_root_dir = None, nms_args = None, flow_type = None, data_type = None):
    model_path = Path(model_path)
    bn = model_path.parent.name
    
    data_type_bn, _, remain = bn.partition('+F')
    if data_type is None:
        data_type = data_type_bn
    
    flow_type_bn, _, remain = remain.partition('+roi')
    if flow_type is None:
        flow_type = flow_type_bn
    
    
    model_name, _, remain = remain.partition('unet-')[-1].partition('_')
    model_name = 'unet-' + model_name
    
    remain = remain.split('_')
    loss_type = remain[0]
    
    if nms_args is None:
        if 'reg' in loss_type:
            nms_args = dict(nms_threshold_abs = 0.4, nms_threshold_rel = None)
        else:
            nms_args = dict(nms_threshold_abs = 0.0, nms_threshold_rel = 0.2)
    
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
                         return_belive_maps = True,
                         **nms_args
                         )
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    if data_root_dir is None:
        data_root_dir = data_args['root_data_dir']
    
    flow_args = flow_types[flow_type]
    
    if '-merged' in data_type:
        flow_func = CoordFlowMerged
    else:
        flow_func = CoordFlow
    
    data_flow = flow_func(data_root_dir / flow_subdir,
                        **flow_args,
                        is_preloaded = False
                        ) 
    
    return model, data_flow, state['epoch']