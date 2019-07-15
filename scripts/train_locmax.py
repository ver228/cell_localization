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
from cell_localization.models import UNet, UNetv2, UNetv2B, UNetscSE, UNetAttention, UNetFlatter, get_loss

from loc_data_types import data_types_dflts
import torch

#%%
def normalize_softmax(xhat):
    n_batch, n_channels, w, h = xhat.shape
    hh = xhat.contiguous().view(n_batch, n_channels, -1)
    hh = torch.nn.functional.softmax(hh, dim = 2)
    hmax, _ = hh.max(dim=2)
    hh = hh/hmax.unsqueeze(2)
    hh = hh.contiguous().view(n_batch, n_channels, w, h)
    return hh

def reshape_norm(xhat):
    n_batch, n_outs, w, h = xhat.shape
    n_channels = n_outs // 5
    hh = xhat.contiguous().view(n_batch, n_channels, 5, w, h)
    hh = hh[:, :, 0]
    hh = normalize_softmax(hh)
    return hh

#%%
    #dd= normalize_softmax(xhat)
    #plt.imshow(dd[0,0].detach().numpy())
#%%
def _get_model(model_name, n_ch_in, n_ch_out):
    batchnorm = '-bn' in model_name
    tanh_head = '-tanh' in model_name
    sigma_out = '-sigmoid' in model_name
    try:
        ll = model_name.split('-')
        ii = ll.index('init')
        init_type = ll[ii+1]
    except ValueError:
        init_type = 'xavier'
    
    
    ss = model_name.partition('-')[0]
    model_args = dict(
            n_channels = n_ch_in, 
             n_classes = n_ch_out, 
             batchnorm = batchnorm,
             tanh_head = tanh_head,
             sigma_out = sigma_out,
             init_type = init_type
            )
    
    if ss == 'unetv2b':
        model = UNetv2B(**model_args)
    elif ss == 'unetv2':
        model = UNetv2(**model_args)
    elif ss == 'unetscSE':
        model = UNetscSE(**model_args)
    elif ss == 'unetattn':
        model = UNetAttention(**model_args)
    
    elif ss == 'unetflatter':
        model = UNetFlatter(**model_args)
    elif ss == 'unet':
        model = UNet(**model_args)
    else:
        raise ValueError('Not implemented `{}`'.format(ss))
        
    return model
    


def train(
        data_type = 'eggsadamv2',
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
        roi_size = 128,
        is_preloaded = False,
        hard_mining_freq = None,
        model_path_init = None,
        **argkws
        ):
    
    dflts = data_types_dflts[data_type]
    dflt_root_data_dir = dflts['root_data_dir']
    flow_args = dflts['flow_args']
    n_ch_in = dflts['n_ch_in']
    n_ch_out = dflts['n_ch_out']
    
    
    if 'maxlikelihood' in loss_type:
        preeval_func = normalize_softmax
        flow_args['loc_gauss_sigma'] = -1
    elif 'mixturemodelloss' in loss_type:
        preeval_func = reshape_norm
        n_ch_out = n_ch_out*5
        flow_args['loc_gauss_sigma'] = -1
    else:
        preeval_func = lambda x : x
    
    if log_dir is None:
        if 'log_prefix' in dflts:
            log_dir = log_dir_root_dflt / 'locmax_detection' / dflts['log_prefix'] / data_type
        else:
            log_dir = log_dir_root_dflt / 'locmax_detection' / data_type
    
    if root_data_dir is None:
        root_data_dir = dflt_root_data_dir
    root_data_dir = Path(root_data_dir)
    
    train_dir = root_data_dir / 'train'
    test_dir = root_data_dir / 'validation'
    
    
    print(root_data_dir)
    train_flow = CoordFlow(train_dir,
                    samples_per_epoch = 40960,
                    roi_size = roi_size,
                    **flow_args,
                    is_preloaded = is_preloaded
                    )  
    
    val_flow = CoordFlow(test_dir,
                    samples_per_epoch = 640,
                    roi_size = roi_size,
                    **flow_args,
                    is_preloaded = is_preloaded
                    ) 
    
    
    model = _get_model(model_name, n_ch_in, n_ch_out)
    if model_path_init is not None:
        model_name += '-pretrained'
        state = torch.load(model_path_init, map_location = 'cpu')
        model.load_state_dict(state['state_dict'])
        
    
    
    device = get_device(cuda_id)
    
    criterion = get_loss(loss_type)
    
    
    
    optimizer = get_optimizer(optimizer_name, 
                              model, 
                              lr = lr, 
                              momentum = momentum, 
                              weight_decay = weight_decay)
    
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    
    hard_mining_str = '' if hard_mining_freq is None else f'_hard-neg-freq{hard_mining_freq}'
    save_prefix = f'{data_type}-roi{roi_size}_{model_name}{hard_mining_str}_{loss_type}_{date_str}'
    save_prefix = f'{save_prefix}_{optimizer_name}_lr{lr}_wd{weight_decay}_batch{batch_size}'
    
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
        preeval_func = preeval_func,
        **argkws
        )
    

if __name__ == '__main__':
    import fire
    fire.Fire(train)
    