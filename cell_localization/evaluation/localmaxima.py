
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:04:14 2019

@author: avelinojaver
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def cv2_peak_local_max(img, threshold_relative, threshold_abs):
    
    #max_val = img.max()
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
    
    th = max(max_val*threshold_relative, threshold_abs)
    
    _, mm = cv2.threshold(img, th, max_val, cv2.THRESH_TOZERO)
    #max filter
    kernel = np.ones((3,3))
    mm_d = cv2.dilate(mm, kernel)
    loc_maxima = cv2.compare(mm, mm_d, cv2.CMP_GE)
    
    mm_e = cv2.erode(mm, kernel)
    non_plateau = cv2.compare(mm, mm_e, cv2.CMP_GT)
    loc_maxima = cv2.bitwise_and(loc_maxima, non_plateau)
    
    
    #the code below is faster than  coords = np.array(np.where(loc_maxima>0)).T
    
    _, coords, _ = cv2.findContours(loc_maxima, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    coords = np.array([x.squeeze() for cc in coords for x in cc])
    coords = np.array(coords)
    
    return coords

def score_coordinates(prediction, target, max_dist = 10, assigment = 'hungarian'):
    if prediction.size == 0:
        return 0, 0, len(target), None, None
    
    if target.size == 0:
        return 0, len(prediction), 0, None, None
    
    cost_matrix = cdist(prediction, target)
    cost_matrix[cost_matrix > max_dist] = max_dist
    
    if assigment == 'hungarian':
        pred_ind, true_ind = linear_sum_assignment(cost_matrix)
        good = cost_matrix[pred_ind, true_ind] < max_dist
        pred_ind, true_ind = pred_ind[good], true_ind[good]
        
    elif assigment == 'greedy':
        #%%
        pred_ind = []
        true_ind = []
        for p_ind, cost_rows in enumerate(cost_matrix):
            t_ind = np.argmin(cost_rows)
            val = cost_rows[t_ind]
            if val <  max_dist and not (t_ind in true_ind):
                true_ind.append(t_ind)
                pred_ind.append(p_ind)
        
        pred_ind = np.array(pred_ind)
        true_ind = np.array(true_ind)
    else:
        raise(f'Not implemented {assigment}')
    
    
    
    TP = pred_ind.size
    FP = prediction.shape[0] - pred_ind.size
    FN = target.shape[0] - pred_ind.size
    
    return TP, FP, FN, pred_ind, true_ind


if __name__ == '__main__':
    pass