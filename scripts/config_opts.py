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
        
        'woundhealing-merged' : {
                'scale_int' : (0, 1.),
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
            },
        
        'lymphocytes' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.25,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1),
            'valid_labels' : [1]
            },
        
        'lymph-eos' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.25,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1),
            'valid_labels' : [1, 2]
            },
        
        'lymphocytesonly' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.0,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1),
            'valid_labels' : [1]
            },
        'eosinophils' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.25,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1),
            'valid_labels' : [2]
            },
        'eosinophilsonly' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.25,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1),
            'valid_labels' : [2]
            },
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
        
        'woundhealing-F0.5-merged': {
        'root_data_dir' : Path.home() / 'workspace/localization/data/woundhealing/annotated/splitted/F0.5x/',
        'log_prefix' : 'woundhealing-F0.5-merged',
        'dflt_flow_type' : 'woundhealing-merged',
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
                
          
        'lymphocytes-20x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/lymphocytes/20x',
        'log_prefix' : 'lymphocytes/20x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
                
        'all-lymphocytes-20x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/all_lymphocytes/20x',
        'log_prefix' : 'lymphocytes/20x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
        
        'lymph-eos-20x' : {
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/lymphocytes/20x',
        'log_prefix' : 'lymphocytes/20x',
        'dflt_flow_type' : 'lymph-eos',
        'n_ch_in' : 3,
        'n_ch_out' : 2
        },
                
        'all-lymph-eos-20x' : {
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/all_lymphocytes/20x',
        'log_prefix' : 'lymphocytes/20x',
        'dflt_flow_type' : 'lymph-eos',
        'n_ch_in' : 3,
        'n_ch_out' : 2
        },
              
                
        'lymphocytes-40x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/lymphocytes/40x',
        'log_prefix' : 'lymphocytes/40x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
                
        'eosinophils-20x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/eosinophils/20x',
        'log_prefix' : 'eosinophils/20x',
        'dflt_flow_type' : 'eosinophils',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
                
                
        'eosinophils-40x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/eosinophils/40x',
        'log_prefix' : 'eosinophils/40x',
        'dflt_flow_type' : 'eosinophils',
        'n_ch_in' : 3,
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
        'unet-simple-bn' : {
             'unet_type' : 'unet-simple',
             'unet_initial_filter_size' : 48, 
             'unet_levels' : 4, 
             'unet_conv_per_level' : 2,
             'unet_increase_factor' : 2,
             'unet_batchnorm' : True,
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
        'unet-flatv2' : {
             'unet_type' : 'unet-simple',
             'unet_initial_filter_size' : 48, 
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
                
        'unet-input-halved' : {
             'unet_type' : 'unet-input-halved',
             'unet_initial_filter_size' : 48, 
             'unet_levels' : 4, 
             'unet_conv_per_level' : 2,
             'unet_increase_factor' : 2,
             'unet_batchnorm' : False,
             'unet_init_type' : 'normal',
             'unet_pad_mode' : 'constant'
             }, 
        'unet-deeper5' : {
             'unet_type' : 'unet-simple',
             'unet_initial_filter_size' : 48, 
             'unet_levels' : 5, 
             'unet_conv_per_level' : 2,
             'unet_increase_factor' : 2,
             'unet_batchnorm' : False,
             'unet_init_type' : 'normal',
             'unet_pad_mode' : 'constant'
             }, 
        
        'unet-deeper6' : {
             'unet_type' : 'unet-simple',
             'unet_initial_filter_size' : 48, 
             'unet_levels' : 6, 
             'unet_conv_per_level' : 2,
             'unet_increase_factor' : 2,
             'unet_batchnorm' : False,
             'unet_init_type' : 'normal',
             'unet_pad_mode' : 'constant'
             }, 
        
        }

