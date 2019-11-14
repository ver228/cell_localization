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

def normalize_softmax(xhat):
    n_batch, n_channels, w, h = xhat.shape
    hh = xhat.contiguous().view(n_batch, n_channels, -1)
    hh = torch.nn.functional.softmax(hh, dim = 2)
    hh = hh.view(n_batch, n_channels, w, h)
    return hh

def get_loss(loss_type):
    
    criterion = None
    if loss_type == 'maxlikelihood':
        criterion = MaximumLikelihoodLoss()
        preevaluation = normalize_softmax
    else:
        parts = loss_type.split('-')
        loss_t = parts[0]
        
        gauss_sigma = [x for x in parts if x[0] == 'G'] #this must exists
        gauss_sigma = float(gauss_sigma[0][1:])
         
        increase_factor = [x for x in parts if x[0] == 'F'] #this one is optional
        increase_factor = float(increase_factor[0][1:]) if increase_factor else 1.
        
        
        is_regularized = [x for x in parts if x == 'reg']
        is_regularized = len(is_regularized) > 0
        
        
        preevaluation = lambda x : x
        if loss_t == 'l1smooth':
            target_loss = nn.SmoothL1Loss()
        elif loss_t == 'l2':
            target_loss = nn.MSELoss()
        elif loss_t == 'l1':
            target_loss = nn.L1Loss()
            
        criterion = LossWithBeliveMaps(target_loss, 
                                       gauss_sigma = gauss_sigma, 
                                       is_regularized = is_regularized,
                                       increase_factor = increase_factor
                                       )
    
    if criterion is None:
        raise ValueError(loss_type)
    return criterion, preevaluation


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
                 mapping_network,
                 loss_type = 'l2-G1.5',
                 
                 nms_threshold_abs = 0.2,
                 nms_threshold_rel = None,
                 nms_min_distance = 3,
                 
                 return_belive_maps = False
                 ):
        
        
        super().__init__()
        
        _dum = set(dir(self))
        
        self.mapping_network_name = mapping_network.__name__
        
        self.n_classes = mapping_network.n_outputs
        self.nms_threshold_abs = nms_threshold_abs
        self.nms_threshold_rel = nms_threshold_rel
        self.nms_min_distance = nms_min_distance
        
        self.loss_type = loss_type
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        self.mapping_network = mapping_network
        
        
        self.criterion, self.preevaluation = get_loss(loss_type)
        self.nms = BeliveMapsNMS(nms_threshold_abs, nms_threshold_rel, nms_min_distance)
        
        self.return_belive_maps = return_belive_maps
    
    
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
        
    
    def forward(self, x, targets = None):
        xhat = self.mapping_network(x)
        outputs = []
        if self.training or (targets is not None):
            loss = dict(
                loss_loc = self.criterion(xhat, targets)
                )
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
            
            