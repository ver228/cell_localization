#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 13:01:35 2019

@author: avelinojaver
"""

import sys
from pathlib import Path 
root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from cell_localization.flow import CoordFlow, collate_simple
from cell_localization.models import get_model
     
import tqdm
from torch.utils.data import DataLoader
import torch
import numpy as np

import matplotlib.pylab as plt

if __name__ == '__main__':

    #%%
    root_dir = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/eosinophils/training/20x/'
    
    num_workers = 4
    batch_size = 4#512
    gauss_sigma = 1.5
    device = 'cpu'
    
    flow_args = dict(
                roi_size = 16,
                scale_int = (0, 4095),
                prob_unseeded_patch = 0.5,
              
                zoom_range = (0.97, 1.03),
                
                int_aug_offset = (-0.2, 0.2),
                int_aug_expansion = (0.7, 1.3),
                samples_per_epoch = batch_size*100
                )
    
    gen = CoordFlow(root_dir, **flow_args)
    
    loader = DataLoader(gen,
                        batch_size = batch_size, 
                        shuffle = True, 
                        num_workers = num_workers,
                        collate_fn = collate_simple
                        )
    
    model = get_model('ind+clf+unet-simple', 3, 2, 'maxlikelihood')
    
    for images, targets in tqdm.tqdm(loader):
        images = torch.from_numpy(np.stack(images)).to(device)
        targets = [{k: torch.from_numpy(v).to(device) for k, v in target.items()} for target in targets]
        
        #%%
        model.train()
        losses = model(images, targets)
        loss = sum([v for v in losses.values()])
        loss.backward()
        
        #%%
        model.eval()
        losses, predictions = model(images, targets)
        break
    #%%
#    import torch.nn.functional as F
#    xhat, features = model.mapping_network(images)
#    
#    
#    
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
#    
#    mm = xhat.detach().numpy()
#    for m, pred in zip(mm, outs):
#        pred_coords = pred[0]
#        boxes = torch.cat((pred_coords - model.proposal_half_size, pred_coords + model.proposal_half_size), dim = -1)
#        proposals.append(boxes) 
#        
#    
#        from matplotlib import patches
#        
#        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize = (10, 10))
#        
#        
#        ax.imshow(m)
#        for box in boxes:
#            cm, w, l = (box[0], box[1]), box[2] - box[0], box[3] - box[1]
#            rect = patches.Rectangle(cm, w, l,linewidth=1,edgecolor='r',facecolor='none')
#            ax.add_patch(rect)
#        break
    
    