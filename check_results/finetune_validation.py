#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 19:55:24 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'scripts') )

import numpy as np
from cell_localization.utils import get_device
from cell_localization.evaluation.localmaxima import score_coordinates
from cell_localization.models import get_model
from cell_localization.flow import CoordFlow

from config_opts import flow_types, data_types

import torch
from pathlib import Path
import tqdm
import matplotlib.pylab as plt
import pickle

def _plot_predictions(image, belive_maps, predictions, target, pred_ind, true_ind):
    img = image.detach().cpu().numpy()
    if image.shape[0] == 3:
        img = np.rollaxis(img, 0, 3)
        img = img[..., ::-1]
    else:
        img = img[0]
    
    mm = belive_maps[0, 0].detach().cpu().numpy()
    
    pred_coords = predictions['coordinates'].detach().cpu().numpy()
    true_coords = target['coordinates'].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize = (15, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Raw Image')
    axs[1].imshow(mm)
    if pred_ind is None:
        axs[1].plot(pred_coords[:, 0], predictions[:, 1], 'x', color = 'r')
        axs[1].plot(target[:, 0], target[:, 1], '.', color = 'r')
        
    else:
        good = np.zeros(pred_coords.shape[0], np.bool)
        good[pred_ind] = True
        pred_bad = pred_coords[~good]
        
        good = np.zeros(true_coords.shape[0], np.bool)
        good[true_ind] = True
        target_bad = true_coords[~good]
        
        axs[0].plot(pred_bad[:, 0], pred_bad[:, 1], 'x', color = 'r')
        axs[0].plot(target_bad[:, 0], target_bad[:, 1], '.', color = 'r')
        axs[0].plot(pred_coords[pred_ind, 0], pred_coords[pred_ind, 1], 'o', color='g')
    

def load_model(model_path, **argkws):
    model_path = Path(model_path)
    bn = model_path.parent.name
    
    data_type, _, remain = bn.partition('+F')
    flow_type, _, remain = remain.partition('+roi')
    model_name, _, remain = remain.partition('_')[-1].partition('_')
    
    remain = remain.split('_')
    loss_type = remain[0]
    
    state = torch.load(model_path, map_location = 'cpu')
    
    data_args = data_types[data_type]
    n_ch_in = data_args['n_ch_in']
    n_ch_out = data_args['n_ch_out']
    
    
    model = get_model(model_name, n_ch_in, n_ch_out, loss_type)
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    return model, state['epoch']


@torch.no_grad()
def calculate_metrics(model, data_flow, device, thresh2check, max_dist):
    N = len(data_flow.data_indexes)
    
    metrics = np.zeros((3, 2, len(thresh2check)))
    for ind in tqdm.trange(N):
        
        image, target = data_flow.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = image.to(device)
    
        predictions = model(image[None])
        predictions = predictions[0]
        
        assert (target['labels'] <= 1).all() #this code should only be used with models with only one label. I worry for the rest later
        
        pred_coords = predictions['coordinates'].detach().cpu().numpy()
        pred_scores_abs = predictions['scores_abs'].detach().cpu().numpy()
        pred_scores_rel = predictions['scores_rel'].detach().cpu().numpy()
        true_coords = target['coordinates'].detach().cpu().numpy()
        
        for ith, th in enumerate(thresh2check):
            for iscore, pred_scores in enumerate((pred_scores_abs, pred_scores_rel)):
                
                pred = pred_coords[pred_scores >= th]
                TP, FP, FN, pred_ind, true_ind = score_coordinates(pred, true_coords, max_dist = max_dist)
                metrics[:, iscore, ith] += (TP, FP, FN)
    
    TP, FP, FN = metrics
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
    
    
    scores = dict(
            TP=TP, 
            FP=FP, 
            FN=FN, 
            P=P, 
            R=R, 
            F1=F1
            )
    
    return scores
    

def get_scores_with_best_threshold(model_path, device, thresh2check, max_dist, train_flow, val_flow):
    model, epoch = load_model(model_path, 
                            nms_threshold_abs = 0.,
                            nms_threshold_rel = 0.,
                            unet_pad_mode = 'reflect'
                            )
    model = model.to(device)
    
    train_scores = calculate_metrics(model, train_flow, device, thresh2check, max_dist)
    
    try:
        ibest = np.nanargmax(train_scores['F1'])
    except ValueError:
        ibest = 0
    
    ith_val = ibest % len(thresh2check)
    ith_type = ibest//len(thresh2check)
    
    th_type = 'abs' if ith_type == 0 else 'rel'
    
    best_threshold = thresh2check[ith_val]
    
    test_scores = calculate_metrics(model, val_flow, device, [best_threshold], max_dist)
    
    test_scores = {k:v[ith_type, 0] for k,v in test_scores.items()}
    
    
    
    res = dict(
            train_scores = train_scores, 
            test_scores = test_scores, 
            best_threshold_type = th_type,
            best_threshold_value = best_threshold, 
            epoch = epoch
            
            )
    
    return res

def main(
    data_type = 'limphocytes-20x',
    flow_type = 'limphocytes',
    
    root_model_dir = Path.home() / 'workspace/localization/results/locmax_detection/limphocytes/20x/limphocytes-20x',
    checkpoint_name = 'model_best.pth.tar',
    
    max_dist = 10,
    thresh2check = np.arange(0.05, 1, 0.05),
    cuda_id = 0,
    ):
    
    device = get_device(cuda_id)
    
    data_args = data_types[data_type]
    root_data_dir = data_args['root_data_dir']
    
    flow_args = flow_types[flow_type]
    
    root_model_dir = Path(root_model_dir)
    train_flow = CoordFlow(root_data_dir / 'train',
                        **flow_args,
                        is_preloaded = True
                        ) 
    val_flow = CoordFlow(root_data_dir / 'validation',
                        **flow_args,
                        is_preloaded = True
                        ) 
    
    
    model_paths = root_model_dir.rglob(checkpoint_name)
    model_paths = list(model_paths)
    
    #model_paths = [x for x in model_paths if 'maxlikelihood' in x.parent.name]
    
    results = {}
    for model_path in tqdm.tqdm(model_paths):
        res = get_scores_with_best_threshold(model_path, device, thresh2check, max_dist, train_flow, val_flow)
        bn = model_path.parent.name
        results[bn] = res
    
    save_name = root_model_dir / 'scores.p'
    with open(save_name, 'wb') as fid:
        pickle.dump(results, fid)

if __name__ == '__main__':
    import fire
    fire.Fire(main)
    