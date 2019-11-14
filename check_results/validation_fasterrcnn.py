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
from matplotlib import patches

if __name__ == '__main__':
   cuda_id = 0
   device = get_device(cuda_id)
   
   model_args = {}
   flow_type = None 
   data_type = None
   
   
   bn = 'crc-det+Fcrc-det-validpatches+roi96_fasterrcnn+resnet18_None_20190821_180313_adam_lr8e-05_wd0.0_batch14'
   model_dir =  Path().home() / 'workspace/localization/results/locmax_detection/CRCHistoPhenotypes/detection/crc-det'
   model_args = dict(nms_min_distance = 5)
   
   data_root_dir = None
   flow_subdir = 'validation' #'train' #
   
   model_path = model_dir / bn / 'model_best.pth.tar'
   model, data_flow, epoch = load_model(model_path, 
                                        data_root_dir = data_root_dir, 
                                        flow_subdir = flow_subdir,
                                        flow_type = flow_type, 
                                        data_type = data_type,
                                        **model_args
                                        )
   model = model.to(device)
   
   #%%
   
   N = len(data_flow.data_indexes)
   
   metrics = np.zeros(3)
   for ind in tqdm.trange(N):
        #%%
        image, target = data_flow.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = image.to(device)
   
        predictions = model(image[None])
        xr = image.squeeze().detach().cpu().numpy()
        pred_coords = predictions[0]['coordinates'].detach().cpu().numpy()
        pred_bboxes = predictions[0]['boxes'].detach().cpu().numpy()
        
        true_coords = target['coordinates'].detach().cpu().numpy()
         
        TP, FP, FN, pred_ind, true_ind = score_coordinates(pred_coords, true_coords, max_dist = 10)
        metrics += TP, FP, FN
        
        img = image.detach().cpu().numpy()
        if image.shape[0] == 3:
            img = np.rollaxis(img, 0, 3)
            img = img[..., ::-1]
        else:
            img = img[0]
        
            
    
        
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize = (15, 5))
        
        ax.imshow(img, cmap='gray')
        ax.set_title('Raw Image')
        if pred_ind is None:
            ax.plot(pred_coords[:, 0], predictions[:, 1], 'x', color = 'r')
            ax.plot(target[:, 0], target[:, 1], '.', color = 'r')
            
        else:
            good = np.zeros(pred_coords.shape[0], np.bool)
            good[pred_ind] = True
            pred_bad = pred_coords[~good]
            
            good = np.zeros(true_coords.shape[0], np.bool)
            good[true_ind] = True
            target_bad = true_coords[~good]
            
            
            for box in pred_bboxes:
                cm, w, l = (box[0], box[1]), box[2] - box[0], box[3] - box[1]
                rect = patches.Rectangle(cm, w, l,linewidth=1,edgecolor='r',facecolor='none')
                ax.add_patch(rect)
            
            ax.plot(pred_bad[:, 0], pred_bad[:, 1], 'x', color = 'r')
            ax.plot(target_bad[:, 0], target_bad[:, 1], '.', color = 'r')
            ax.plot(pred_coords[pred_ind, 0], pred_coords[pred_ind, 1], 'o', color='g')
            
            
            
    #%%
   TP, FP, FN = metrics
   P = TP/(TP+FP)
   R = TP/(TP+FN)
   F1 = 2*P*R/(P+R)
   
   print(f'P={P} | R={R} | F1={F1}')
   
   #%%
   
     
     