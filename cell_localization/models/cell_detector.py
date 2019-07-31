#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 17:46:38 2019

@author: avelinojaver
"""
import torch
from torch import nn
import torch.nn.functional as F

from .losses import MaximumLikelihoodLoss, LossWithBeliveMaps
from .unet import unet_constructor, unet_attention, unet_squeeze_excitation, unet_input_halved

def normalize_softmax(xhat):
    n_batch, n_channels, w, h = xhat.shape
    hh = xhat.contiguous().view(n_batch, n_channels, -1)
    hh = torch.nn.functional.softmax(hh, dim = 2)
    hh = hh.view(n_batch, n_channels, w, h)
    return hh

def get_loss(loss_type):
    
    loss_t, _, gauss_sigma = loss_type.partition('-G')
    loss_t, _, is_regularized = loss_t.partition('-')
    is_regularized = is_regularized == 'reg'
    
    
    criterion = None
    if gauss_sigma:
        gauss_sigma = float(gauss_sigma)
    
    if loss_t == 'maxlikelihood':
        criterion = MaximumLikelihoodLoss()
        preevaluation = normalize_softmax
    else:
        preevaluation = lambda x : x
        if loss_t == 'l1smooth':
            target_loss = nn.SmoothL1Loss()
        elif loss_t == 'l2':
            target_loss = nn.MSELoss()
        elif loss_t == 'l1':
            target_loss = nn.L1Loss()
            
        criterion = LossWithBeliveMaps(target_loss, 
                                       gauss_sigma = gauss_sigma, 
                                       is_regularized = is_regularized
                                       )
    
    if criterion is None:
        raise ValueError(loss_type)
    
    return criterion, preevaluation

def get_unet_model(model_type, n_inputs, n_ouputs,  **argkws):
    
    if model_type == 'unet-simple':
        constructor = unet_constructor
    elif model_type == 'unet-attention':
        constructor = unet_attention 
    elif model_type == 'unet-SE':
        constructor = unet_squeeze_excitation
    elif model_type == 'unet-input-halved':
        constructor = unet_input_halved
    else:
        raise ValueError(f'Not implemented {model_type}')
    
    model = constructor(n_inputs, n_ouputs, **argkws)
    return model

class BeliveMapsNMS(nn.Module):
    def __init__(self, threshold_abs = 0.0, threshold_rel = None, min_distance = 3):
        super().__init__()
        self.threshold_abs = threshold_abs
        self.threshold_rel = threshold_rel
        self.min_distance = min_distance
        
    def forward(self, belive_map):
        kernel_size = 2 * self.min_distance + 1
        
        n_batch, n_channels, w, h = belive_map.shape
        hh = belive_map.contiguous().view(n_batch, n_channels, -1)
        max_vals, _ = hh.max(dim=2)
        
        x_max = F.max_pool2d(belive_map, kernel_size, stride = 1, padding = kernel_size//2)
        x_mask = (x_max == belive_map) #nms using local maxima filtering
        
        
        x_mask &= (belive_map > self.threshold_abs) 
        if self.threshold_rel is not None:
            vmax = max_vals.view(n_batch, n_channels, 1, 1)
            x_mask &= (belive_map > self.threshold_rel*vmax)
        
        
        outputs = []
        for xi, xm, xmax in zip(belive_map, x_mask, max_vals):
            ind = xm.nonzero()
            scores_abs = xi[ind[:, 0], ind[:, 1], ind[:, 2]]
            scores_rel = scores_abs/xmax[ind[:, 0]]
            
            
            labels = ind[:, 0] + 1 #add one since I am naming the classes from 1 onwards
            coodinates = ind[:, [2, 1]]
            outputs.append((coodinates, labels, scores_abs, scores_rel))
        
        return outputs

class CellDetector(nn.Module):
    def __init__(self, 
                 unet_type = 'unet-simple',
                 unet_n_inputs = 1,
                 unet_n_ouputs = 1,
                 unet_initial_filter_size = 48, 
                 unet_levels = 4, 
                 unet_conv_per_level = 2,
                 unet_increase_factor = 2,
                 unet_batchnorm = False,
                 unet_init_type = 'xavier',
                 unet_pad_mode = 'constant',
                 
                 loss_type = 'l2-G1.5',
                 
                 nms_threshold_abs = 0.2,
                 nms_threshold_rel = None,
                 nms_min_distance = 3,
                 
                 return_belive_maps = False
                 ):
        super().__init__()
        
        self.n_classes = unet_n_ouputs
        
        self.mapping_network = get_unet_model(model_type = unet_type, 
                                    n_inputs = unet_n_inputs,
                                    n_ouputs = unet_n_ouputs,
                                    initial_filter_size = unet_initial_filter_size, 
                                    levels = unet_levels, 
                                    conv_per_level = unet_conv_per_level,
                                    increase_factor = unet_increase_factor,
                                    batchnorm = unet_batchnorm,
                                    init_type = unet_init_type,
                                    pad_mode = unet_pad_mode
                                   )
        
        
        self.criterion, self.preevaluation = get_loss(loss_type)
        self.nms = BeliveMapsNMS(nms_threshold_abs, nms_threshold_rel, nms_min_distance)
        
        self.return_belive_maps = return_belive_maps

    def forward(self, x, targets = None):
        xhat = self.mapping_network(x)
         
        outputs = []
        if self.training or (targets is not None):
            loss = self.criterion(xhat, targets)
            outputs.append(loss)
        
        
        if not self.training:
            result = []
            xhat = self.preevaluation(xhat)
            outs = self.nms(xhat)
            
            for coordinates, labels, scores_abs, scores_rel in outs:
                assert (labels > 0).all()
                
                result.append(
                    dict(
                        coordinates = coordinates,
                        labels = labels,
                        scores_abs = scores_abs,
                        scores_rel = scores_rel
                        )
                    )
            outputs.append(result)

        if self.return_belive_maps:
            outputs.append(xhat)
        
        if len(outputs) == 1:
            outputs = outputs[0]
        
        return outputs
            
            