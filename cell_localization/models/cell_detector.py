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
from .unet import unet_constructor, unet_attention, unet_squeeze_excitation

def normalize_softmax(xhat):
    n_batch, n_channels, w, h = xhat.shape
    hh = xhat.view(n_batch, n_channels, -1)
    hh = torch.nn.functional.softmax(hh, dim = 2)
    hmax, _ = hh.max(dim=2)
    hh = hh/hmax.unsqueeze(2)
    hh = hh.view(n_batch, n_channels, w, h)
    return hh

def get_loss(loss_type):
    loss_type, _, gauss_sigma = loss_type.partition('-G')
    loss_type, _, is_regularized = loss_type.partition('-')
    is_regularized = is_regularized == 'reg'
    
    if gauss_sigma:
        gauss_sigma = float(gauss_sigma)
    
    preevaluation = lambda x : x
    if loss_type == 'l1smooth':
        criterion = LossWithBeliveMaps(nn.SmoothL1Loss(), gauss_sigma, is_regularized)
    
    elif loss_type == 'l2':
        criterion = LossWithBeliveMaps(nn.MSELoss(), gauss_sigma, is_regularized)
        
    elif loss_type == 'maxlikelihood':
        criterion = MaximumLikelihoodLoss()
        preevaluation = normalize_softmax
    else:
        raise ValueError(loss_type)
    
    return criterion, preevaluation

def get_unet_model(model_type, n_inputs, n_ouputs,  **argkws):
    
    if model_type == 'unet-simple':
        constructor = unet_constructor
    elif model_type == 'unet-attention':
        constructor = unet_attention 
    elif model_type == 'unet-SE':
        constructor = unet_squeeze_excitation
    else:
        raise ValueError(f'Not implemented {model_type}')
    
    model = constructor(n_inputs, n_ouputs, **argkws)
    return model

class BeliveMapsNMS(nn.Module):
    def __init__(self, threshold_abs = 0.5, min_distance = 3, exclude_border = True):
        super().__init__()
        self.threshold_abs = threshold_abs
        self.min_distance = min_distance
        self.exclude_border = exclude_border
    
    def forward(self, belive_map):
        kernel_size = 2 * self.min_distance + 1
        
        x_max = F.max_pool2d(belive_map, kernel_size, stride = 1, padding = kernel_size//2)
        x_mask = (x_max == belive_map) #nms using local maxima filtering
        
        if self.exclude_border:
            exclude = 2 * self.min_distance
            x_mask[..., :exclude] = x_mask[..., -exclude:] = 0
            x_mask[..., :exclude, :] = x_mask[..., -exclude:, :] = 0
        x_mask &= belive_map > self.threshold_abs
        
        
        outputs = []
        for xi, xm in zip(belive_map, x_mask):
            ind = xm.nonzero()
            scores = xi[ind[:, 0], ind[:, 1], ind[:, 2]]
            labels = ind[:, 0] + 1 #add one since I am naming the classes from 1 onwards
            coodinates = ind[:, [2, 1]]
            outputs.append((coodinates, labels, scores))
        
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
                 nms_min_distance = 3,
                 nms_exclude_border = True,
                 
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
        self.nms = BeliveMapsNMS(nms_threshold_abs, nms_min_distance, nms_exclude_border)
        
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
            
            for coordinates, labels, scores in outs:
                assert (labels > 0).all()
                
                result.append(
                    dict(
                        coordinates = coordinates,
                        labels = labels,
                        scores = scores
                        )
                    )
            outputs.append(result)

        if self.return_belive_maps:
            outputs.append(xhat)
        
        if len(outputs) == 1:
            outputs = outputs[0]
        
        return outputs
            
            