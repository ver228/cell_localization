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


from load_model import load_model


import tqdm
import matplotlib.pylab as plt

if __name__ == '__main__':
    cuda_id = 0
    device = get_device(cuda_id)
    
    model_args = {}
    flow_type = None 
    data_type = None
   
   #bn = 'worm-eggs-adam+Feggs+roi96_unet-simple_l2-G1.5_20190717_162118_adam_lr0.000128_wd0.0_batch128'
   #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-1_unet-simple_l2-G1.5_20190717_162710_adam_lr0.000128_wd0.0_batch128'
   
   #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/eggs'
   
    #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
    #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/eggs'
    
    bn = 'worm-eggs-adam-masks+Feggs+roi128+hard-neg-5_unet-simple_maxlikelihood_20190808_151948_adam_lr0.000128_wd0.0_batch64'
    model_dir = Path.home() / '/Users/avelinojaver/workspace/localization/results/locmax_detection/eggs/worm-eggs-adam-masks'
    
   #bn = 'woundhealing-v2-nuclei+Fwoundhealing+roi96_unet-simple_maxlikelihood_20190718_221552_adam_lr0.000128_wd0.0_batch128'
   #bn = 'woundhealing-v2-mix+Fwoundhealing+roi96_unet-simple_maxlikelihood_20190718_222136_adam_lr0.000128_wd0.0_batch128'
   #bn = 'woundhealing-v2-mix+Fwoundhealing+roi96_unet-simple_l2-reg-G2.5_20190719_145851_adam_lr0.000128_wd0.0_batch128'
   #bn = 'woundhealing-v2-nuclei+Fwoundhealing+roi48_unet-simple_l1-reg-G1.5_20190719_194158_adam_lr0.000256_wd0.0_batch256'
   
    #bn = 'woundhealing-v2-mix/woundhealing-v2-mix+Fwoundhealing+roi96_unet-simple_l2-reg-G1.5_20190726_135131_adam_lr0.000128_wd0.0_batch128'
    #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/'
   
    #bn = 'woundhealing-v2-mix/woundhealing-v2-mix+Fwoundhealing+roi96_unet-simple_l2-reg-G1.5_20190726_135131_adam_lr0.000128_wd0.0_batch128'
    #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/'
   

#    bn = 'woundhealing-v2-mix+Fwoundhealing+roi48_unet-simple_maxlikelihood_20190719_021908_adam_lr0.000256_wd0.0_batch256'
#    model_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-v2/woundhealing-v2-mix/different_losses_complete/roi48/'
#    model_args = dict(nms_min_distance = 5, nms_threshold_rel = 0.1, nms_threshold_abs=0.0)
#    
    
    #bn = 'woundhealing-F0.5-merged/woundhealing-F0.5-merged+Fwoundhealing-merged+roi48_unet-simple_l2-G2.5_20190730_162214_adam_lr0.000256_wd0.0_batch256'
    #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing-F0.5-merged/'
    #data_type = 'woundhealing-v2-mix'
    #flow_type = 'woundhealing' 
   
   
   #bn = 'eosinophils-20x+Feosinophils+roi64_unet-simple_maxlikelihood_20190723_233841_adam_lr0.000256_wd0.0_batch256'
   
   #bn = 'eosinophils-20x/eosinophils-20x+Feosinophils+roi48_unet-simple_l2-reg-G1.5_20190726_141052_adam_lr0.000256_wd0.0_batch256'
   #bn = 'eosinophils-20x/maxlikelihood/eosinophils-20x+Feosinophils+roi48_unet-simple_maxlikelihood_20190725_074506_adam_lr0.000256_wd0.0_batch256'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/eosinophils/20x/'
   
   #bn = 'lymphocytes-20x/lymphocytes-20x+Flymphocytes+roi48_unet-simple_l2-reg-G1.5_20190729_190056_adam_lr0.000256_wd0.0_batch256'
   #bn = 'lymphocytes-20x/lymphocytes-20x+Flymphocytes+roi48_unet-simple-bn_maxlikelihood_20190729_190738_adam_lr0.000256_wd0.0_batch256'
   
   #bn = 'lymphocytes-20x/lymphocytes-20x+Flymphocytes+roi48_unet-simple_l2-G2.5_20190729_233108_adam_lr0.000256_wd0.0_batch256'
   #bn = 'all-lymphocytes-20x/all-lymphocytes-20x+Flymphocytes+roi96_unet-simple_maxlikelihood_20190730_025548_adam_lr0.000128_wd0.0_batch128'
   #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/lymphocytes/20x/'

   #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190802_161236_adam_lr0.000128_wd0.0_batch128'#'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
   #model_dir =  Path().home() / 'workspace/localization/results/locmax_detection/eggs/worm-eggs-adam/'


    #bn = 'lymphocytes-40x/lymphocytes-40x+Flymphocytes+roi96_unet-attention_l2-G2.5_20190803_104752_adam_lr9.6e-05_wd0.0_batch96'
    #model_dir = Path.home() / 'workspace/localization/results/locmax_detection/lymphocytes/40x/'
  
    #bn = 'crc-det+Fcrc-det+roi96_clf+unet-simple_maxlikelihood_20190821_050952_adam_lr0.000128_wd0.0_batch128'
    #bn = 'crc-det+Fcrc-det+roi96_clf+unet-simple_l2-G2.5_20190820_221608_adam_lr0.000128_wd0.0_batch128'
    #bn = 'crc-det+Fcrc-det+roi96_unet-simple_l2-G2.5_20190821_014053_adam_lr0.000128_wd0.0_batch128'
    #bn = 'crc-det+Fcrc-det+roi96_unet-simple_maxlikelihood_20190821_084438_adam_lr0.000128_wd0.0_batch128'
    
    #bn = 'crc-det+Fcrc-det+roi96_ind+clf+unet-simple_maxlikelihood_20190823_120311_adam_lr0.000128_wd0.0_batch128'
    #model_dir =  Path().home() / 'workspace/localization/results/locmax_detection/CRCHistoPhenotypes/detection/crc-det'
   
   
    #model_dir =  Path().home() / 'workspace/localization/results/locmax_detection/CRCHistoPhenotypes/classification/crc-clf'
    #bn = 'crc-clf+Fcrc-clf+roi96_ind+clf+unet-simple_maxlikelihood_20190822_215819_adam_lr9.6e-05_wd0.0_batch96'
    #model_args = dict(nms_min_distance = 5)
   
    #data_root_dir = Path.home() / 'workspace/localization/data/worm_eggs_first' 
    data_root_dir = None
   
   
    model_path = model_dir / bn / 'model_best.pth.tar'
    #model_path = model_dir / bn / 'checkpoint.pth.tar'
    
    flow_subdir = 'validation'
    #flow_subdir = 'train'
   
    model, data_flow, epoch = load_model(model_path, 
                                         flow_subdir = flow_subdir,
                                        data_root_dir = data_root_dir, 
                                        flow_type = flow_type, 
                                        data_type = data_type, 
                                        return_belive_maps = True,
                                        **model_args
                                        )
    model = model.to(device)
   
   #%%
   
    N = len(data_flow.data_indexes)
   
    metrics = np.zeros(3)
    #for ind in tqdm.trange(N):
    for ind in [25]:   
        image, target = data_flow.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = image.to(device)
   
        predictions, belive_maps = model(image[None])
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
        
        if isinstance(belive_maps, tuple):
            belive_maps = belive_maps[0]
            
        
        mm = belive_maps[0, 0].detach().cpu().numpy()
        #%%
        #figsize = (50, 120)
        figsize = (40, 5)
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize = figsize)
        
        axs[0].imshow(img, cmap='gray')
        axs[1].imshow(mm)
        axs[2].imshow(img, cmap='gray')
        #axs[0].set_title('Raw Image')
        
        for ax in axs:
            ax.axis('off')
        
        if pred_ind is None:
            axs[2].plot(pred_coords[:, 0], pred_coords[:, 1], 'x', color = 'r')
            axs[2].plot(target[:, 0], target[:, 1], '.', color = 'r')
            
        else:
            good = np.zeros(pred_coords.shape[0], np.bool)
            good[pred_ind] = True
            pred_bad = pred_coords[~good]
            
            good = np.zeros(true_coords.shape[0], np.bool)
            good[true_ind] = True
            target_bad = true_coords[~good]
            
            axs[2].plot(pred_bad[:, 0], pred_bad[:, 1], 'x', color = 'r')
            axs[2].plot(target_bad[:, 0], target_bad[:, 1], '.', color = 'r')
            axs[2].plot(pred_coords[pred_ind, 0], pred_coords[pred_ind, 1], 'o', color='g')
        
        #fig.savefig('worms.pdf', bbox_inches = 'tight')
        #plt.axis([850, 1050, 1200, 1400])
        #fig.savefig('worms_zoomed.pdf', bbox_inches = 'tight')
        #plt.axis([450, 550, 1225, 1325])
        #fig.savefig('worms_zoomedv2.pdf', bbox_inches = 'tight')
        
        #fig.savefig('woundhealing.pdf', bbox_inches = 'tight')
        #plt.axis([0, 100, 0, 100])
        #fig.savefig('woundhealing_zoomed.pdf', bbox_inches = 'tight')
        
        
        #%%
#    #%%
#    import torch
#    import torch.nn.functional as F
#    from torchvision.ops import roi_align
#    
#    xhat, features = model.mapping_network(image[None])
#    #I want to get a map to indicate if there is an cell or not
#    feats = features[0].permute((0, 2, 3, 1))
#    n_batch, clf_h, clf_w, clf_n_filts = feats.shape
#    feats = feats.contiguous().view(-1, clf_n_filts, 1, 1)
#    clf_scores = model.clf_patch_head(feats)
#    #scores, has_cells = clf_scores.max(dim=1)
#    clf_scores = F.softmax(clf_scores, dim = 1)            
#    clf_scores = clf_scores[:, 1].view(n_batch, 1, clf_h, clf_w)
#    
#    
#    clf_scores = F.interpolate(clf_scores, size = xhat.shape[-2:], mode = 'bilinear', align_corners=False)
#    
#    
#    bad = clf_scores< 0.5
#    xhat[bad] = xhat[bad].mean()
#    xhat = model.preevaluation(xhat)
#    outs = model.nms(xhat)
#    #%%
#    proposals = []
#    target_labels = []
#    for pred, true in zip(outs, [target]):
#        pred_coords = pred[0]
#        
#        offsets = pred_coords.view(-1, 1, 2) - true['coordinates'].view(1, -1, 2)
#        dists2 = (offsets**2).sum(dim=-1)
#        min_dist2, match_id = dists2.min(dim=1)
#        
#        
#        r2_lim = model.proposal_match_r2
#        r2_lim = 49
#        valid_matches = min_dist2 <= r2_lim # I am using the squre distanc to avoid having to cast to float
#        
#        n_pred = len(pred_coords)
#        labels = torch.zeros(n_pred, dtype = torch.long, device=pred_coords.device, requires_grad = False)
#        labels[valid_matches] = true['labels'][match_id[valid_matches]]
#        target_labels.append(labels)
#        
#        boxes = torch.cat((pred_coords - model.proposal_half_size, pred_coords + model.proposal_half_size), dim = -1)
#        proposals.append(boxes) 
#    #%%
#    
#    plt.figure(figsize = (10, 10))
#    plt.imshow(img)
#    
#    
#    ii = 1
#    
#    good = labels == 1
#    bad = labels == 0
#    plt.plot(pred_coords[good,0], pred_coords[good,1], 'og')
#    plt.plot(pred_coords[bad,0], pred_coords[bad,1], 'oc')
#    
#    plt.plot(target['coordinates'][:,0], target['coordinates'][:,1], 'xr')
#    
#    #%%
#    
#    
#    
#    #%%
#    from matplotlib import patches
#    
#    for boxes in proposals:
#        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize = (10, 10))
#        #ax.imshow(m[0])
#        ax.imshow(img)
#        for box in boxes:
#            cm, w, l = (box[0], box[1]), box[2] - box[0], box[3] - box[1]
#            rect = patches.Rectangle(cm, w, l,linewidth=1,edgecolor='r',facecolor='none')
#            ax.add_patch(rect)
#    #%%
#    proposals = [x.float() for x in proposals] # I need to optimize this 
#    pooled_feats = roi_align(features[-1], proposals, model.roi_pool_size, 1)
#    labels_v = model.clf_proposal_head(pooled_feats)
#    ff = roi_align(image[None], proposals, (15, 15), 1)
#    ff = pooled_feats.permute(0, 2, 3, 1).detach().numpy()
#    plt.figure()
#    plt.imshow(ff[2, ..., ::-1])
    #%%

    
    
    TP, FP, FN = metrics
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    F1 = 2*P*R/(P+R)
   
    print(f'P={P} | R={R} | F1={F1}')
     
   #%%