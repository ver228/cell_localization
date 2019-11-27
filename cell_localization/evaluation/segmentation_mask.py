
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: avelinojaver
"""

import cv2
import numpy as np

def get_masks_metrics(true_cells_mask, pred_cells_mask):
    
    n_true, true_seg_mask, true_stats, true_centroids = cv2.connectedComponentsWithStats(true_cells_mask, 4)
    n_pred, pred_seg_mask, pred_stats, pred_centroids = cv2.connectedComponentsWithStats(pred_cells_mask, 4)
    target_coords = true_centroids[1:]
    true_areas = true_stats[1:, -1]
   
    pred_coords = pred_centroids[1:]
    pred_areas = pred_stats[1:, -1]
    
     
    if not len(pred_coords) or not len(pred_coords):
         return pred_coords, target_coords, np.zeros((len(target_coords), len(pred_coords))), 0, np.sum(pred_areas) + np.sum(pred_areas)
    
    
    inds = true_seg_mask*n_pred + pred_seg_mask
    counts = np.bincount(inds.flatten(), minlength = n_true*n_pred)
    intersections = counts.reshape((n_true, n_pred))[1:, 1:]
    
    unions = true_areas[:, None] + pred_areas[None, :] - intersections
    
    IoU = intersections/unions
    
    #Jaccard aggregated index. Algorithm 1 from:
    #"A Dataset and a Technique for Generalized Nuclear Segmentation for Computational Pathology"
    
    used_inds = []
    aggregated_union = 0
    aggregated_intersection = 0
    
    for union, intersection in zip(unions, intersections):
        best_match = np.argmax(intersection)
        if intersection[best_match] > 0:
            used_inds.append(best_match)
        aggregated_union += union[best_match]
        aggregated_intersection += intersection[best_match]
        
        
        
    unused_inds = set(range(unions.shape[1])) - set(used_inds)
    for ss in unused_inds:
        aggregated_union += pred_areas[ss]
    
    
    
    return pred_coords, target_coords, IoU, aggregated_intersection, aggregated_union


def get_IoU_best_match(IoU, match_threshold = 0.1):
    
    if IoU.shape[1] == 0:
        return 0, 0, IoU.shape[0], np.zeros(0), np.zeros(0)
    
    if IoU.shape[0] == 0:
        return 0, IoU.shape[1], 0, np.zeros(0), np.zeros(0)
    
    #I am expecting IoU to have dimenssions [n_targets, n_predictions]
    true_ind = np.arange(IoU.shape[0])
    pred_ind = np.argmax(IoU, axis = 1)
    IoU_matches = IoU[true_ind, pred_ind]
    
    ind_sorted = np.argsort(IoU_matches)[::-1]
    IoU_matches, pred_ind, true_ind = [x[ind_sorted] for x in (IoU_matches, pred_ind, true_ind)]
    
    dat = []
    for val, t, p in zip( IoU_matches, true_ind, pred_ind):
        if val < match_threshold:
            break
        
        if dat:
            _, trues, preds = zip(*dat)
            if t in trues or p in preds:
                continue
            
        dat.append((val, t, p))
    
    if not dat:
        return 0, IoU.shape[1], 0, None, None
    
    assert all([len(dat) <= x for x in IoU.shape])
    IoU_matches, true_ind, pred_ind = map(np.array, zip(*dat))
    
    TP = len(true_ind)
    FN = IoU.shape[1] - TP
    FP = IoU.shape[0] - TP
    
    return TP, FP, FN, pred_ind, true_ind

def segmentation2contours(seg_mask, kernel_size = 5, chain_approx = False):
    chain_approx = cv2.CHAIN_APPROX_SIMPLE  if chain_approx else cv2.CHAIN_APPROX_NONE
    
    n_labs, seg_mask, stats, centroids = cv2.connectedComponentsWithStats(seg_mask.astype(np.uint8), 4)
        
    kernel = cv2.getStructuringElement( cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    mm = kernel_size // 2 #offset
    
    cell_contours = []
    for ii, (top, left, height, width, _) in enumerate(stats):
        if ii == 0:
            continue
        
        yl, yr = max(0, left - mm), min(seg_mask.shape[0], left + width + mm) 
        xl, xr = max(0, top - mm), min(seg_mask.shape[1], top + height + mm) 
        
        crop = cv2.compare(seg_mask[yl:yr, xl:xr], ii, cv2.CMP_EQ)
        crop = cv2.dilate(crop, kernel)
        cnt = cv2.findContours(crop, cv2.RETR_EXTERNAL, chain_approx)[-2]
        assert len(cnt) == 1
        
        cnt = cnt[0].squeeze(1) + np.array((xl, yl))[None]
        
        cell_contours.append(cnt)
    
    return cell_contours

if __name__ == '__main__':
    pass