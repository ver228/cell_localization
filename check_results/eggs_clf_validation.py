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

import torch
import torch.nn.functional as F
import numpy as np
from cell_localization.trainer import get_device
from cell_localization.evaluation.localmaxima import score_coordinates
from cell_localization.models import CellDetectorWithClassifier
from cell_localization.flow import CoordFlow

from config_opts import flow_types, data_types, model_types


import tqdm
import matplotlib.pylab as plt


def load_model(model_path, flow_subdir = 'validation', data_root_dir = None, nms_args = None, flow_type = None, data_type = None):
    model_path = Path(model_path)
    bn = model_path.parent.name
    
    data_type_bn, _, remain = bn.partition('+F')
    if data_type is None:
        data_type = data_type_bn
    
    flow_type_bn, _, remain = remain.partition('+roi')
    if flow_type is None:
        flow_type = flow_type_bn
    
    
    model_name, _, remain = remain.partition('unet-')[-1].partition('_')
    model_name = 'unet-' + model_name
    
    remain = remain.split('_')
    loss_type = remain[0]
    
    if nms_args is None:
        if 'reg' in loss_type:
            nms_args = dict(nms_threshold_abs = 0.4, nms_threshold_rel = None)
        else:
            nms_args = dict(nms_threshold_abs = 0.0, nms_threshold_rel = 0.2)
    
    state = torch.load(model_path, map_location = 'cpu')
    
    data_args = data_types[data_type]
    n_ch_in = data_args['n_ch_in']
    n_ch_out = data_args['n_ch_out']
    
    model_args = model_types[model_name]
    model_args['unet_pad_mode'] = 'reflect'
    
    model = CellDetectorWithClassifier(**model_args, 
                         unet_n_inputs = n_ch_in, 
                         unet_n_ouputs = n_ch_out,
                         loss_type = loss_type,
                         return_belive_maps = True,
                         **nms_args
                         )
    
    model.load_state_dict(state['state_dict'])
    model.eval()
    
    if data_root_dir is None:
        data_root_dir = data_args['root_data_dir']
    
    flow_args = flow_types[flow_type]
    
    data_flow = CoordFlow(data_root_dir / flow_subdir,
                        **flow_args,
                        is_preloaded = False
                        ) 
    
    return model, data_flow, state['epoch']

@torch.no_grad()
def _predictions_from_crops(image):
    #image crops
    crops_l = []
    crop_size = 96
    for ii in range(0, image.shape[1],  crop_size - 32):
        for jj in range(0, image.shape[2],  crop_size - 32):
            if ii +  crop_size >  image.shape[1]:
                ii = image.shape[1] - crop_size
            if jj +  crop_size >  image.shape[2]:
                jj = image.shape[2] - crop_size
            
            
            corner = (jj, ii)
            roi = image[:, ii:ii+crop_size, jj:jj+crop_size]
            crops_l.append((corner, roi))
    assert all([x[-1].shape[-2:] == (96, 96)  for x in crops_l])
    
    corners, crops = zip(*crops_l)
    crops = torch.stack(crops)
    corners = torch.tensor(corners, device = crops.device)
    
    xhat, features = model.mapping_network(crops)
    clf_scores = model.clf_head(features[0])
    
    score, has_cells = clf_scores.max(dim =1)
    good = has_cells == 1
    xhat_valid = xhat[good]
    corners_valid = corners[good]
    
    xhat_valid = model.preevaluation(xhat_valid)
    
    
    clf_img = torch.zeros((1, 1, *image.shape[1:3]))
    xhat_pooled = torch.zeros((1, 1, *image.shape[1:3]))
    for corner, xhat_crop in zip(corners_valid, xhat_valid):
        j,i = corner
        clf_img[0, :, i:i+crop_size, j:j+crop_size] += 1.
        xhat_pooled[0, :, i:i+crop_size, j:j+crop_size] += xhat_crop/xhat_crop.max()
        
    xhat_pooled = xhat_pooled/clf_img
    xhat_pooled[clf_img == 0] = 0
    
    outs = model.nms(xhat_pooled)
    
    predictions = []
    for coordinates, labels, scores_abs, scores_rel in outs:
        res = dict(
                    coordinates = coordinates,
                    labels = labels,
                    scores_abs = scores_abs,
                    scores_rel = scores_rel,
                    
                    )
        predictions.append(res)  
    return predictions
#        outs = model.nms(xhat_valid)
#        
#        coordinates, labels, scores_abs, scores_rel = zip(*outs)
#        
#        #inplace offset
#        for corner, coord in zip(corners_valid, coordinates):
#            coord += corner[None, :]
#          
#        coordinates, labels, scores_abs, scores_rel = map(torch.cat, [coordinates, labels, scores_abs, scores_rel])
#        
#        
#        predictions = [
#                    dict(
#                        coordinates = coordinates,
#                        labels = labels,
#                        scores_abs = scores_abs,
#                        scores_rel = scores_rel,
#                        
#                        )
#                    ]
          
    

if __name__ == '__main__':
   cuda_id = 0
   device = get_device(cuda_id)
   flow_type = None 
   data_type = None
   
   
   bn = 'lymphocytes-20x+Flymphocytes+roi96+hard-neg-1_clf+unet-simple_maxlikelihood_20190804_101105_adam_lr0.000128_wd0.0_batch128'
   #bn = 'worm-eggs-adam+Feggs+roi96+hard-neg-5_clf+unet-simple_maxlikelihood_20190804_095849_adam_lr0.000128_wd0.0_batch128'
   
   model_dir = Path('/Users/avelinojaver/OneDrive - Nexus365/loc_with_clf/')
   
   #bn = 'lymphocytes-20x/lymphocytes-20x+Flymphocytes+roi96+hard-neg-1_clf+unet-simple_maxlikelihood_20190804_101105_adam_lr0.000128_wd0.0_batch128'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/lymphocytes/20x/'


   #bn = 'lymphocytes-40x/lymphocytes-40x+Flymphocytes+roi96_unet-attention_l2-G2.5_20190803_104752_adam_lr9.6e-05_wd0.0_batch96'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/lymphocytes/40x/'
  
   #data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs_first' 
   data_root_dir = None
   
   
   model_path = model_dir / bn / 'model_best.pth.tar'
   model, data_flow, epoch = load_model(model_path, data_root_dir = data_root_dir, flow_type = flow_type, data_type = data_type)
   model = model.to(device)

   #%%
   
   N = len(data_flow.data_indexes)
   
   metrics = np.zeros(3)
   for ind in [27]:#tqdm.trange(N):
        
        image, target = data_flow.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = image.to(device)
        
        #full image
        with torch.no_grad():
            xhat, features = model.mapping_network(image[None])
            
            feats = features[0].permute((0, 2, 3, 1))
            n_batch, h, w, n_filts = feats.shape
            feats = feats.view(-1, n_filts, 1, 1)
            
            clf_scores = model.clf_head(feats)
            #scores, has_cells = clf_scores.max(dim=1)
            clf_scores = F.softmax(clf_scores, dim = 1)            
            clf_scores = clf_scores[:, 1].view(1, 1, h, w)
            clf_scores = F.interpolate(clf_scores, size=image.shape[-2:], mode = 'bilinear', align_corners=False)
            
            xhat_v = model.preevaluation(xhat)
            xhat_v[clf_scores< 0.5] = 0
            
            outs = model.nms(xhat_v)
            
            predictions = []
            for coordinates, labels, scores_abs, scores_rel in outs:
                res = dict(
                            coordinates = coordinates,
                            labels = labels,
                            scores_abs = scores_abs,
                            scores_rel = scores_rel,
                            
                            )
                predictions.append(res)
        
        xr = image.squeeze().detach().cpu().numpy()
        pred_coords = predictions[0]['coordinates'].detach().cpu().numpy()
        true_coords = target['coordinates'].detach().cpu().numpy()
         
        TP, FP, FN, pred_ind, true_ind = score_coordinates(pred_coords, true_coords, max_dist = 10)
        metrics += TP, FP, FN
        
        img = image.detach().cpu().numpy()
        if image.shape[0] == 3:
            img = np.rollaxis(img, 0, 3)
            img = img[..., ::-1]
        else:
            img = img[0]
    
       # mm = belive_maps[0, 0].detach().cpu().numpy()
        
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize = (15, 5))
        
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title('Raw Image')
        #axs[1].imshow(clf_img)
        #axs[1].imshow(mm)
        
        mm = xhat_v[0, 0].detach()
        mm/= mm.max()
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
            
            axs[1].plot(pred_coords[:, 0], pred_coords[:, 1], 'x', color = 'r')
            axs[1].plot(true_coords[:, 0], true_coords[:, 1], '.', color = 'c')
            
  
   TP, FP, FN = metrics
   P = TP/(TP+FP)
   R = TP/(TP+FN)
   F1 = 2*P*R/(P+R)
   
   print(f'P={P} | R={R} | F1={F1}')
     
     