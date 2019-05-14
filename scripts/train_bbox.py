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


import torch
import datetime

from cell_localization.trainer import train_bboxes, get_device, log_dir_root_dflt
from cell_localization.flow import CoordFlow, BoxEncoder
from cell_localization.models import RetinaNet, get_loss

data_types_dflts = {
    'heba': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/heba/data',
        min_radius = 5,
        roi_size = 96,
        scale_int = (0, 4095),
        num_classes = 1
        ),
    'eggs': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/worm_eggs',
        min_radius = 2,
        roi_size = 96,
        scale_int = (0, 255),
        num_classes = 1
        ),
            
    'bladder-20x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/20x',
        min_radius = 3,
        roi_size = 96,
        scale_int = (0, 255),
        num_classes = 2
        ),
    
    'bladder-40x': dict(
        root_data_dir = Path.home() / 'workspace/localization/data/histology_bladder/bladder_cancer_tils/rois/40x',
        min_radius = 5,
        roi_size = 128,
        scale_int = (0, 255),
        num_classes = 2
        ) 
        
    }

def train(
        data_type = 'heba',
        model_name = 'retinanet-resnet34',
        loss_type = 'focal',
        cuda_id = 0,
        log_dir = None,
        batch_size = 16,
        num_workers = 1,
        root_data_dir = None,
        lr = 1e-5,
        **argkws
        ):
    
    if log_dir is None:
        log_dir = log_dir_root_dflt / 'bbox_detection' / data_type
    
    
    dflts = data_types_dflts[data_type]
    root_data_dir = dflts['root_data_dir']
    roi_size = dflts['roi_size']
    min_radius = dflts['min_radius']
    scale_int = dflts['scale_int']
    num_classes = dflts['num_classes']
    
    if root_data_dir is None:
        root_data_dir = root_data_dir
    
    train_dir = root_data_dir / 'train'
    test_dir = root_data_dir / 'validation'
    
    bbox_encoder = BoxEncoder(img_size = (roi_size, roi_size),
               pyramid_levels = [1, 2, 3, 4, 5],
               aspect_ratios = [(1.,1.)]
               )
    
    print(root_data_dir)
    train_flow = CoordFlow(train_dir,
                    samples_per_epoch = 20480,
                    bbox_encoder = bbox_encoder,
                    min_radius = min_radius,
                    roi_size = roi_size,
                    scale_int = scale_int
                    )  
    
    val_flow = CoordFlow(test_dir,
                    samples_per_epoch = 640,
                    bbox_encoder = bbox_encoder,
                    min_radius = min_radius,
                    roi_size = roi_size,
                    scale_int = scale_int
                    ) 
    
    if model_name.startswith('retinanet'):
        backbone = model_name.split('-')[1]
    
        model = RetinaNet(backbone = backbone, 
                 is_fpn_small = True,
                 num_classes = num_classes, 
                 num_anchors = 3)
    
    device = get_device(cuda_id)
    
    loss_type = f'{loss_type}-{num_classes}'
    criterion = get_loss(loss_type)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_prefix = f'{data_type}_{model_name}_{loss_type}_{date_str}'
    save_prefix = f'{save_prefix}_adam_lr{lr}_batch{batch_size}'
    
    train_bboxes(save_prefix,
        model,
        device,
        train_flow,
        val_flow,
        criterion,
        optimizer,
        log_dir = log_dir,
        batch_size = batch_size,
        num_workers = num_workers,
        **argkws
        )
    

if __name__ == '__main__':
    import fire
    
    fire.Fire(train)
    