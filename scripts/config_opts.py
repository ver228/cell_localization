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
        
        'woundhealing-contour' : {
            'scale_int' : (0, 4095),
            'zoom_range' : (0.5, 1.1),
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

        
        'lymphocytes-noaug' : {
            'scale_int' : (0, 255),
            'zoom_range' : (1.0, 1.0),
            'prob_unseeded_patch' : 0.25,
            'int_aug_offset' : None,
            'int_aug_expansion' : None,
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

        'crc-det' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.25,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1),
            'valid_labels' : [1]
            },
        'crc-det-validpatches' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.0,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1),
            'valid_labels' : [1]
            },
        'crc-clf' : {
            'scale_int' : (0, 255),
            'zoom_range' : (0.90, 1.1),
            'prob_unseeded_patch' : 0.25,
            'int_aug_offset' : (-0.15, 0.15),
            'int_aug_expansion' : (0.9, 1.1),
            'valid_labels' : [1, 2, 3, 4]
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
                
        'woundhealing-contour': {
        'root_data_dir' : Path.home() / 'workspace/segmentation/data/wound_area_masks.hdf5',
        'log_prefix' : 'woundhealing-contour',
        'dflt_flow_type' : 'woundhealing-contour',
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
            
        'worm-eggs-adam-masks': {
        'root_data_dir' : Path.home() / 'workspace/localization/data/worm_eggs_with_masks',
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

        'crc-det':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/CRCHistoPhenotypes/detection/',
        'log_prefix' : 'CRCHistoPhenotypes/detection',
        'dflt_flow_type' : 'crc-det',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },

        'crc-clf':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/CRCHistoPhenotypes/classification/',
        'log_prefix' : 'CRCHistoPhenotypes/classification',
        'dflt_flow_type' : 'crc-clf',
        'n_ch_in' : 3,
        'n_ch_out' : 4
        },
                
        'TMA-lymphocytes-40x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_data/40x_TMA_lymphocytes_2Bfirst104868.hdf5',
        'log_prefix' : 'lymphocytes/40x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
                
        'TMA-lymphocytes-20x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_data/20x-resized_TMA_lymphocytes_2Bfirst104868.hdf5',
        'log_prefix' : 'lymphocytes/20x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
                
        'MoNuSeg-40x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_data/40x_MoNuSeg_training.hdf5',
        'log_prefix' : 'MoNuSeg/40x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
        'Weinert2012-20x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_data/20x_Weinert_2012.hdf5',
        'log_prefix' : 'Weinert2012/20x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
        'PSB2015-Pathologists-40x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_data/40x_PSB_2015_Pathologists.hdf5',
        'log_prefix' : 'PSB_2015_Pathologists/40x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
        'Naylor-TNBC-40x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_data/40x_Naylor_TNBC.hdf5',
        'log_prefix' : 'Naylor_TNBC/40x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
        'CRCHistoPhenotypes-Det-20x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_data/20x_CRCHistoPhenotypes_2016_04_28_Detection.hdf5',
        'log_prefix' : 'CRCHistoPhenotypes/Detection/20x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
                
        'andrewjanowczyk-nuclei-40x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_data/40x_andrewjanowczyk_nuclei.hdf5',
        'log_prefix' : 'andrewjanowczyk/nuclei/40x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
        'andrewjanowczyk-lymphocytes-40x':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/histology_data/40x_andrewjanowczyk_lymphocytes.hdf5',
        'log_prefix' : 'andrewjanowczyk/lymphocytes/40x',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 3,
        'n_ch_out' : 1
        },
                
        'BBBC038-fluorescence':{
        'root_data_dir' : Path.home() / 'workspace/localization/data/BBBC038_Kaggle_2018_Data_Science_Bowl//fluorescence/',
        'log_prefix' : 'BBBC038/fluorescence',
        'dflt_flow_type' : 'lymphocytes',
        'n_ch_in' : 1,
        'n_ch_out' : 1
        },
                
    }