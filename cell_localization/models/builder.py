#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:27:31 2018

@author: avelinojaver
"""
from .cell_detector_with_clf import CellDetectorWithClassifier
from .cell_detector_with_clf_v2 import CellDetectorWithClassifierInd
from .cell_segmentator import CellSegmentator
from .cell_detector import CellDetector
from .detector_fasterrcnn import FasterRCNNFixedSize
from .unet import get_mapping_network

from functools import partial

def get_model(model_name, 
              n_ch_in, 
              n_ch_out, 
              loss_type, 
              nms_threshold_abs = None, 
              nms_threshold_rel = None, 
              nms_min_distance = 3,
              return_belive_maps = False, 
              **argkws
              ):
    
    if model_name.startswith('fasterrcnn'):
        backbone_name = model_name.split('+')[1]
        model = FasterRCNNFixedSize(n_classes = n_ch_out, backbone_name = backbone_name)
    
    else:
        
        if ((nms_threshold_abs is None) and (nms_threshold_rel is None)):
            if 'reg' in loss_type:
                nms_threshold_abs = 0.4
                nms_threshold_rel = 0.0
            else:
                nms_threshold_abs = 0.0 
                nms_threshold_rel = 0.2
        
        detector_args = dict(
                nms_threshold_abs = nms_threshold_abs,
                nms_threshold_rel = nms_threshold_rel,
                nms_min_distance = nms_min_distance
                )
        
        if model_name.startswith('seg+'):
            model_obj = CellSegmentator 
            model_name = model_name[4:]
            return_feat_maps = False
            detector_args = {}
            
        elif model_name.startswith('clf+'):
            model_obj = CellDetectorWithClassifier 
            model_name = model_name[4:]
            return_feat_maps = True
        
        elif model_name.startswith('ind+clf+'):
            model_obj = partial(CellDetectorWithClassifierInd, n_classes = n_ch_out)
            n_ch_out = 1
            
            model_name = model_name[8:]
            return_feat_maps = True
            
            nms_threshold_abs = 0.0 
            nms_threshold_rel = 0.025
        
        else:
            model_obj = CellDetector
            return_feat_maps = False
        
        
        
        model_args = model_types[model_name].copy()
        model_args.update(argkws)
        model_args['return_feat_maps'] = return_feat_maps
        mapping_network = get_mapping_network(n_ch_in, n_ch_out, **model_args)
        model = model_obj(mapping_network, 
                             loss_type = loss_type,
                             return_belive_maps = return_belive_maps,
                             **detector_args
                             )
    
    return model

model_types = {
        'unet-simple' : {
             'model_type' : 'unet-simple',
             'initial_filter_size' : 48, 
             'levels' : 4, 
             'conv_per_level' : 2,
             'increase_factor' : 2,
             'batchnorm' : False,
             'init_type' : None,
             'pad_mode' : 'constant'
             },
        'unet-simple-bn' : {
             'model_type' : 'unet-simple',
             'initial_filter_size' : 48, 
             'levels' : 4, 
             'conv_per_level' : 2,
             'increase_factor' : 2,
             'batchnorm' : True,
             'init_type' : None,
             'pad_mode' : 'constant'
             },
         'unet-attention' : {
             'model_type' : 'unet-attention',
             'initial_filter_size' : 48, 
             'levels' : 4, 
             'conv_per_level' : 2,
             'increase_factor' : 2,
             'batchnorm' : False,
             'init_type' : None,
             'pad_mode' : 'constant'
             },
        'unet-SE' : {
             'model_type' : 'unet-SE',
             'initial_filter_size' : 48, 
             'levels' : 4, 
             'conv_per_level' : 2,
             'increase_factor' : 2,
             'batchnorm' : False,
             'init_type' : None,
             'pad_mode' : 'constant'
             }, 
        'unet-flat-96' : {
             'model_type' : 'unet-simple',
             'initial_filter_size' : 96, 
             'levels' : 4, 
             'conv_per_level' : 2,
             'increase_factor' : 1,
             'batchnorm' : False,
             'init_type' : None,
             'pad_mode' : 'constant'
             }, 
        'unet-flat-48' : {
             'model_type' : 'unet-simple',
             'initial_filter_size' : 48, 
             'levels' : 4, 
             'conv_per_level' : 2,
             'increase_factor' : 1,
             'batchnorm' : False,
             'init_type' : None,
             'pad_mode' : 'constant'
             }, 
        'unet-wide' : {
             'model_type' : 'unet-simple',
             'initial_filter_size' : 48, 
             'levels' : 2, 
             'conv_per_level' : 2,
             'increase_factor' : 4,
             'batchnorm' : False,
             'init_type' : None,
             'pad_mode' : 'constant'
             }, 
                
        'unet-input-halved' : {
             'model_type' : 'unet-input-halved',
             'initial_filter_size' : 48, 
             'levels' : 4, 
             'conv_per_level' : 2,
             'increase_factor' : 2,
             'batchnorm' : False,
             'init_type' : None,
             'pad_mode' : 'constant'
             }, 
        'unet-deeper5' : {
             'model_type' : 'unet-simple',
             'initial_filter_size' : 48, 
             'levels' : 5, 
             'conv_per_level' : 2,
             'increase_factor' : 2,
             'batchnorm' : False,
             'init_type' : None,
             'pad_mode' : 'constant'
             }, 
        
        'unet-deeper6' : {
             'model_type' : 'unet-simple',
             'initial_filter_size' : 48, 
             'levels' : 6, 
             'conv_per_level' : 2,
             'increase_factor' : 2,
             'batchnorm' : False,
             'init_type' : None,
             'pad_mode' : 'constant'
             }, 
        
         'edsr-r16x64' : {
             'model_type' : 'EDSR',
             'n_resblocks' : 16, 
             'n_feats' : 64,
             'res_scale' : 1.
             },
         
         'edsr-r32x256' : {
             'model_type' : 'EDSR',
             'n_resblocks' : 32, 
             'n_feats' : 256,
             'res_scale' : 0.1
             },
                 
        'unet-resnet18' : {
             'model_type' : 'resnet',
             'backbone_name' : 'resnet18'
             },
        'unet-resnet34' : {
             'model_type' : 'resnet',
             'backbone_name' : 'resnet34'
             },
        'unet-resnet50' : {
             'model_type' : 'resnet',
             'backbone_name' : 'resnet50'
             },
        'unet-resnet101' : {
             'model_type' : 'resnet',
             'backbone_name' : 'resnet101'
             },
        'unet-resnet152' : {
             'model_type' : 'resnet',
             'backbone_name' : 'resnet152'
             },
        'unet-resnext101' : {
             'model_type' : 'resnet',
             'backbone_name' : 'resnext101_32x8d'
             }, 
                
        'unet-densenet121' : { 
                'model_type' : 'densenet',
                'backbone_name' : 'densenet121',
                    },
        'unet-densenet201' : { 
                'model_type' : 'densenet',
                'backbone_name' : 'densenet201',
                    },
                
        'dense-unet' : { 
                'model_type' : 'dense-unet',
                'backbone_name' : 'densenet121',
                    },
        'unet-n2n' : {
             'model_type' : 'unet-n2n'
             #'init_type' : 'xavier',
             #'pad_mode' : 'reflect'
             },
        }