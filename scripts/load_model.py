#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:29:14 2019

@author: avelinojaver
"""
#%%

from config_opts import flow_types, data_types

from cell_localization.models import get_model
from cell_localization.flow import CoordFlow

import torch
from pathlib import Path

def load_model(model_path, 
               flow_subdir = 'validation', 
               data_root_dir = None, 
               flow_type = None, 
               data_type = None,
               **argkws
               ):
    model_path = Path(model_path)
    bn = model_path.parent.name
    
    data_type_bn, _, remain = bn.partition('+F')
    flow_type_bn, _, remain = remain.partition('+roi')
    model_name, _, remain = remain.partition('_')[-1].partition('_')
    
    remain = remain.split('_')
    loss_type = remain[0]
    
    
    if data_type is None:
        data_type = data_type_bn
    
    if flow_type is None:
        flow_type = flow_type_bn
    
    flow_args = flow_types[flow_type]
    data_args = data_types[data_type]
    
    if data_root_dir is None:
        data_root_dir = data_args['root_data_dir']
    state = torch.load(model_path, map_location = 'cpu')
    
    n_ch_in = data_args['n_ch_in']
    n_ch_out = data_args['n_ch_out']
    model = get_model(model_name, n_ch_in, n_ch_out, loss_type, **argkws)
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    data_flow = CoordFlow(data_root_dir / flow_subdir,
                        **flow_args,
                        is_preloaded = False
                        ) 
    
    return model, data_flow, state['epoch']