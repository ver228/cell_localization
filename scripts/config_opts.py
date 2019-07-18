#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:29:14 2019

@author: avelinojaver
"""
from pathlib import Path

flow_types = {
        'woundhealing' : {
            'scale_int' : (0, 4095),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.2,
            'int_aug_offset' : (-0.2, 0.2),
            'int_aug_expansion' : (0.5, 1.3)
            },
        'eggs' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.5,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1)
            },
        'eggsonly' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.0,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1)
            }
        }

data_types = {
        'woundhealing-v2-mix': {
        'root_data_dir' : Path.home() / 'workspace/localization/data/woundhealing/annotated/v2/mix',
        'log_prefix' : 'woundhealing-v2',
        'dflt_flow_type' : 'woundhealing',
         'n_ch_in'  : 1,
         'n_ch_out' : 1
        },
                
        'woundhealing-v2-nuclei': {
        'root_data_dir' : Path.home() / 'workspace/localization/data/woundhealing/annotated/v2/nuclei',
        'log_prefix' : 'woundhealing-v2',
        'dflt_flow_type' : 'woundhealing',
         'n_ch_in'  : 1,
         'n_ch_out' : 1
        },
                
        'worm-eggs-adam': {
        'root_data_dir' : Path.home() / 'workspace/localization/data/worm_eggs_adam',
        'log_prefix' : 'eggs',
        'dflt_flow_type' : 'eggs',
         'n_ch_in'  : 1,
         'n_ch_out' : 1
        },
          
    }

model_types = {
        'unet-simple' : {
             'unet_type' : 'unet-simple',
             'unet_initial_filter_size' : 48, 
             'unet_levels' : 4, 
             'unet_conv_per_level' : 2,
             'unet_increase_factor' : 2,
             'unet_batchnorm' : False,
             'unet_init_type' : 'normal',
             'unet_pad_mode' : 'constant'
             },
         'unet-attention' : {
             'unet_type' : 'unet-attention',
             'unet_initial_filter_size' : 48, 
             'unet_levels' : 4, 
             'unet_conv_per_level' : 2,
             'unet_increase_factor' : 2,
             'unet_batchnorm' : False,
             'unet_init_type' : 'normal',
             'unet_pad_mode' : 'constant'
             },
        'unet-SE' : {
             'unet_type' : 'unet-SE',
             'unet_initial_filter_size' : 48, 
             'unet_levels' : 4, 
             'unet_conv_per_level' : 2,
             'unet_increase_factor' : 2,
             'unet_batchnorm' : False,
             'unet_init_type' : 'normal',
             'unet_pad_mode' : 'constant'
             }, 
        'unet-flat' : {
             'unet_type' : 'unet-simple',
             'unet_initial_filter_size' : 96, 
             'unet_levels' : 4, 
             'unet_conv_per_level' : 2,
             'unet_increase_factor' : 1,
             'unet_batchnorm' : False,
             'unet_init_type' : 'normal',
             'unet_pad_mode' : 'constant'
             }, 
        'unet-wide' : {
             'unet_type' : 'unet-simple',
             'unet_initial_filter_size' : 48, 
             'unet_levels' : 2, 
             'unet_conv_per_level' : 2,
             'unet_increase_factor' : 4,
             'unet_batchnorm' : False,
             'unet_init_type' : 'normal',
             'unet_pad_mode' : 'constant'
             }, 
        
        }

