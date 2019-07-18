#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 17:05:49 2018œ

@author: avelinojaver
"""

from pathlib import Path 

data_types_dflts = {
        'woundhealing-v2-nuclei': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v2/nuclei',
        log_prefix = 'woundhealing-v2',
        flow_args = dict(
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
        'woundhealing-v2-mix': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v2/mix',
        log_prefix = 'woundhealing-v2',
        flow_args = dict(
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
        'woundhealing-v2-mix+nuclei': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v2/mix+nuclei',
        log_prefix = 'woundhealing-v2',
        flow_args = dict(
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
        
        
        'eggsadam': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam',
        log_prefix = 'eggs',
        flow_args = dict(
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 1.5,
                int_aug_offset = (-0.15, 0.15),
                int_aug_expansion = (0.85, 1.2)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    
    'eggsadamv2': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam',
        log_prefix = 'eggs',
        flow_args = dict(
                scale_int = (0, 255),
                prob_unseeded_patch = 0.25,
                loc_gauss_sigma = 1.5,
                zoom_range = (0.97, 1.03),
                ignore_borders = True,
                min_radius = 2.,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.7, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    
    'eggsadamI': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam',
        log_prefix = 'eggs',
        flow_args = dict(
                scale_int = (0, 255),
                prob_unseeded_patch = 0.25,
                loc_gauss_sigma = -1,
                zoom_range = (0.97, 1.03),
                ignore_borders = True,
                min_radius = 2.,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.7, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    
    'eggsadamv3': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam',
        log_prefix = 'eggs',
        flow_args = dict(
                scale_int = (0, 255),
                prob_unseeded_patch = 0.25,
                loc_gauss_sigma = 1.5,
                zoom_range = (0.97, 1.03),
                ignore_borders = False,
                min_radius = 2.,
                int_aug_offset = (-0.01, 0.01),
                int_aug_expansion = (0.95, 1.05)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    

    'eggsadam-stacked': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam',
        log_prefix = 'eggs',
        flow_args = dict(
                scale_int = (0, 255),
                stack_shape = (4,4),
                prob_unseeded_patch = 0.25,
                loc_gauss_sigma = -1,
                zoom_range = (0.97, 1.03),
                ignore_borders = False,
                min_radius = 2.,
                int_aug_offset = (-0.01, 0.01),
                int_aug_expansion = (0.95, 1.05)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    
    'eggsadam-stacked3': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam',
        log_prefix = 'eggs',
        flow_args = dict(
                scale_int = (0, 255),
                stack_shape = (3,3),
                prob_unseeded_patch = 0.25,
                loc_gauss_sigma = -1,
                zoom_range = (0.97, 1.03),
                ignore_borders = False,
                min_radius = 2.,
                int_aug_offset = (-0.01, 0.01),
                int_aug_expansion = (0.95, 1.05)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
        
    'eggsadamrefined': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs_adam_refined',
        log_prefix = 'eggs',
        flow_args = dict(
                scale_int = (0, 255),
                prob_unseeded_patch = 0.25,
                loc_gauss_sigma = 1.5,
                zoom_range = (0.97, 1.03)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    
    
    
    'bladder-tiles-20x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/20x',
        log_prefix = 'bladder/20x',
        flow_args = dict(
            scale_int = (0, 255),
            prob_unseeded_patch = 0.5,
            loc_gauss_sigma = 2.5
            ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
    
    'bladder-tiles-40x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/40x',
        log_prefix = 'bladder/40x',
        flow_args = dict(
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 5
                ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
                
    'bladder-tiles-roi96-20x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/20x',
        log_prefix = 'bladder/20x',
        flow_args = dict(
            roi_size = 96,
            scale_int = (0, 255),
            prob_unseeded_patch = 0.5,
            loc_gauss_sigma = 2.5
            ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
    'bladder-tiles-roi64-20x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/20x',
        log_prefix = 'bladder/20x',
        flow_args = dict(
            roi_size = 64,
            scale_int = (0, 255),
            prob_unseeded_patch = 0.5,
            loc_gauss_sigma = 2.5
            ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
    'bladder-tiles-roi48-20x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/20x',
        log_prefix = 'bladder/20x',
        flow_args = dict(
            roi_size = 48,
            scale_int = (0, 255),
            prob_unseeded_patch = 0.5,
            loc_gauss_sigma = 2.5
            ),
        n_ch_in = 3,
        n_ch_out = 2
        ),
    
    'bladder-tiles-roi32-20x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/20x',
        log_prefix = 'bladder/20x',
        flow_args = dict(
            roi_size = 32,
            scale_int = (0, 255),
            prob_unseeded_patch = 0.5,
            loc_gauss_sigma = 2.5
            ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
                
    
    'bladder-tiles-roi128-40x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/40x',
        log_prefix = 'bladder/40x',
        flow_args = dict(
            roi_size = 128,
            scale_int = (0, 255),
            prob_unseeded_patch = 0.5,
            loc_gauss_sigma = 2.5
            ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
    'bladder-tiles-roi64-40x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/40x',
        log_prefix = 'bladder/40x',
        flow_args = dict(
            roi_size = 64,
            scale_int = (0, 255),
            prob_unseeded_patch = 0.5,
            loc_gauss_sigma = 2.5
            ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
                
    'bladder-tiles-roi48-40x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/40x',
        log_prefix = 'bladder/40x',
        flow_args = dict(
            roi_size = 48,
            scale_int = (0, 255),
            prob_unseeded_patch = 0.5,
            loc_gauss_sigma = 2.5
            ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
    'bladder-tiles-no-border-40x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/40x',
        log_prefix = 'bladder',
        flow_args = dict(
                roi_size = 96,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 5
                ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ) ,
    'bladder-tiles-no-border-20x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/full_tiles/20x',
        log_prefix = 'bladder',
        flow_args = dict(
                roi_size = 96,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 5
                ),
        n_ch_in = 3,
        n_ch_out = 2
        
        ) 
        
    }


data_types_old = {
    'woundhealing-no-membrane-roi48': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/no_membrane',
        log_prefix = 'woundhealing',
        flow_args = dict(
                roi_size = 48,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'woundhealing-no-membrane-roi96': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/no_membrane',
        log_prefix = 'woundhealing',
        flow_args = dict(
                roi_size = 96,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'woundhealing-only-membrane-roi48': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/only_membrane',
        log_prefix = 'woundhealing',
        flow_args = dict(
                roi_size = 48,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    
    'woundhealing-only-membrane-roi96': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/only_membrane',
        log_prefix = 'woundhealing',
        flow_args = dict(
                roi_size = 96,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
        
    'woundhealing-all': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/all',
        log_prefix = 'woundhealing',
        flow_args = dict(
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
        
    'woundhealing-all-roi48': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/annotated/v1/all',
        log_prefix = 'woundhealing',
        flow_args = dict(
                roi_size = 48,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    
    'woundhealing-demixed-roi48': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/woundhealing/demixed_predictions',
        log_prefix = 'woundhealing',
        flow_args = dict(
                roi_size = 48,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.25, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
                
    'heba': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 96,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-int': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 64,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-tv0': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data-v0',
        flow_args = dict(
                roi_size = 64,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-tuncorrected': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data-uncorrected',
        flow_args = dict(
                roi_size = 64,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-v0-int': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 64,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-v0': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 64,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-v0-int-patchnorm': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 64,
                prob_unseeded_patch = 0.2,
                loc_gauss_sigma = 2,
                patchnorm = True,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-v0-patchnorm': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 64,
                prob_unseeded_patch = 0.2,
                loc_gauss_sigma = 2,
                patchnorm = True
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
                
    
    'heba-uncorrected': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 64,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-uncorrected-int': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 64,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
        
    'heba-uncorrected-int-roi32': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 32,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-uncorrected-int-roi96': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 32,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'heba-uncorrected-int-roi128': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        flow_args = dict(
                roi_size = 128,
                prob_unseeded_patch = 0.2,
                scale_int = (0, 4095),
                loc_gauss_sigma = 2,
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.5, 1.3)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
        
    'eggs': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs',
        flow_args = dict(
                roi_size = 64,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 1.5
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'eggs-int': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs',
        flow_args = dict(
                roi_size = 64,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 1.5,
                int_aug_offset = (-0.15, 0.15),
                int_aug_expansion = (0.85, 1.2)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'eggs-only': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs',
        flow_args = dict(
                roi_size = 64,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.0,
                loc_gauss_sigma = 1.5,
                int_aug_offset = (-0.15, 0.15),
                int_aug_expansion = (0.85, 1.2)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'eggs-int-old': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/old_worm_eggs',
        flow_args = dict(
                roi_size = 64,
                scale_int = (0, 255),
                prob_unseeded_patch = 0.5,
                loc_gauss_sigma = 1.5,
                int_aug_offset = (-0.15, 0.15),
                int_aug_expansion = (0.85, 1.2)
                ),
        n_ch_in = 1,
        n_ch_out = 1
        ),
    'bladder-20x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/20x',
        log_prefix = 'bladder',
        roi_size = 64,
        scale_int = (0, 255),
        prob_unseeded_patch = 0.0,
        loc_gauss_sigma = 2.5,
        n_ch_in = 3,
        n_ch_out = 2
        
        ),
    
    'bladder-40x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/40x',
        log_prefix = 'bladder',
        roi_size = 128,
        scale_int = (0, 255),
        prob_unseeded_patch = 0.0,
        loc_gauss_sigma = 5,
        n_ch_in = 3,
        n_ch_out = 2
        ),
        
        
        
        }