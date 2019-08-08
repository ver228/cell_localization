#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:41:15 2019

@author: avelinojaver
"""
import torch
from torch import nn
import torch.nn.functional as F

from .cell_detector import BeliveMapsNMS, get_unet_model, get_loss
from .unet import ConvBlock

class SimpleClassifier(nn.Sequential):
    def __init__(self, n_inputs, n_classes, dropout_p = 0.2):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        
        self.conv = ConvBlock(n_inputs, 128)
        self.pool = nn.AdaptiveMaxPool2d(1)
        
        self.clf = nn.Sequential(
                nn.Linear(128, 32),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Linear(32, n_classes),
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
                )
        
    def forward(self, xin):
        x = self.conv(xin)
        x = self.pool(x)
        
        x = x.view(-1, 128)
        xout = self.clf(x)
        return xout

class CellDetectorWithClassifier(nn.Module):
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
        
        _dum = set(dir(self))
        
        self.unet_type = unet_type
        self.unet_n_inputs = unet_n_inputs
        self.unet_n_ouputs = unet_n_ouputs
        self.unet_initial_filter_size = unet_initial_filter_size
        self.unet_levels = unet_levels
        self.unet_conv_per_level = unet_conv_per_level
        self.unet_increase_factor = unet_increase_factor
        self.unet_batchnorm = unet_batchnorm
        self.unet_init_type = unet_init_type
        self.unet_pad_mode = unet_pad_mode
        
        self.nms_threshold_abs = nms_threshold_abs
        self.nms_threshold_rel = nms_threshold_rel
        self.nms_min_distance = nms_min_distance
        
        self.loss_type = loss_type
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        
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
                                    pad_mode = unet_pad_mode,
                                    return_feat_maps = True
                                   )
        
        
        self.criterion_loc, self.preevaluation = get_loss(loss_type)
        self.nms = BeliveMapsNMS(nms_threshold_abs, nms_threshold_rel, nms_min_distance)
        
        self.return_belive_maps = return_belive_maps
        
        n_filters_clf = self.mapping_network.down_blocks[-1].n_filters[-1]
        self.clf_head = SimpleClassifier(n_filters_clf, 2)
        self.criterion_clf = nn.CrossEntropyLoss()
    
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
        
    
    def forward(self, x, targets = None):
        xhat, features = self.mapping_network(x)
        
        
        outputs = []
        if self.training or (targets is not None):
            clf_scores = self.clf_head(features[0])
            valid_ = [t['labels'].shape[0]>0 for t in targets]
            valid_ = torch.tensor(valid_, device = xhat.device)
            
            valid_loc_xhat = xhat[valid_]
            valid_loc_targets = [t for t in targets if t['labels'].shape[0]>0]
            clf_target = valid_.long()
            
            loss = dict(
                loss_loc = self.criterion_loc(valid_loc_xhat, valid_loc_targets),
                loss_clf = self.criterion_clf(clf_scores, clf_target)
                )
            
            outputs.append(loss)
        
        
        if not self.training:
            #I want to get a map to indicate if there is an egg or not
            feats = features[0].permute((0, 2, 3, 1))
            n_batch, clf_h, clf_w, clf_n_filts = feats.shape
            feats = feats.view(-1, clf_n_filts, 1, 1)
            clf_scores = self.clf_head(feats)
            #scores, has_cells = clf_scores.max(dim=1)
            clf_scores = F.softmax(clf_scores, dim = 1)            
            clf_scores = clf_scores[:, 1].view(n_batch, 1, clf_h, clf_w)
            
            
            clf_scores = F.interpolate(clf_scores, size = xhat.shape[-2:], mode = 'bilinear', align_corners=False)
            xhat_v = self.preevaluation(xhat)
            xhat_v[clf_scores< 0.5] = 0
            
            outs = self.nms(xhat_v)
            
            predictions = []
            for coordinates, labels, scores_abs, scores_rel in outs:
                res = dict(
                            coordinates = coordinates,
                            labels = labels,
                            scores_abs = scores_abs,
                            scores_rel = scores_rel,
                            
                            )
                predictions.append(res)
            
            outputs.append(predictions)

        if self.return_belive_maps:
            outputs.append(xhat)
        
        if len(outputs) == 1:
            outputs = outputs[0]
        
        return outputs


            