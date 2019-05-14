#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 13:13:13 2018

@author: avelinojaver
"""
import math
import numpy as np

def get_anchor_boxes(
        img_size = (256, 256),
        box_ini_scale = 2, #this value stablish how the size of the box to regress increases with the pyramid level
        pyramid_levels = [3, 4, 5, 6, 7],
        aspect_ratios = [(1.,1.)],   #here I am only using 1-1 ratio because I am using only circles in the dataset
        scales = [1., 2**(1/3), 2**(2/3)],
        ):
    
    #boxes sizes, they do not have to match the cell receptive field. 
    #I choice this values because I am expecting an output of 256
    box_sizes = [int(box_ini_scale*(2 ** x)) for x in pyramid_levels]
    
    
    grid_sizes = [[int(math.ceil(cc/2**x)) for cc in img_size] for x in pyramid_levels]
    strides = [2 ** x for x in pyramid_levels] #how the boxes will be shifted
    
    n_shapes = len(scales)*len(aspect_ratios)
    
    all_anchors = []
    for (grid_size_y, grid_size_x), stride, box_size in zip(grid_sizes, strides, box_sizes):
        anchors_shape = [(sc*arx*box_size,sc*ary*box_size) for (arx, ary) in aspect_ratios for sc in scales]
        
        anchors_shape = np.array(anchors_shape)
        
        
        shift_x = (np.arange(0, grid_size_x) + 0.5) * stride
        shift_y = (np.arange(0, grid_size_y) + 0.5) * stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        anchor_coords = np.stack((shift_x.flatten(), shift_y.flatten()), axis=1)
        n_shifts = anchor_coords.shape[0]
        
        anchor_coords = anchor_coords[:, None, :].repeat(repeats = n_shapes, axis=1)
        anchors_shape = anchors_shape[None].repeat(repeats = n_shifts, axis=0)
        
        level_anchors = np.concatenate((anchor_coords, anchors_shape), axis=2)
        level_anchors = level_anchors.reshape(n_shifts, -1)
        
        all_anchors.append(level_anchors)
        
    all_anchors = np.concatenate(all_anchors)
    all_anchors = all_anchors.reshape(-1, 4)
    
    return all_anchors, n_shapes

def wh2xy(bbox):
    #change format from xc,yc,w h -> x1, y1,x2, y2
    x0 = bbox[...,0] - bbox[...,2]/2
    x1 = bbox[...,0] + bbox[...,2]/2
    y0 = bbox[...,1] - bbox[...,3]/2
    y1 = bbox[...,1] + bbox[...,3]/2
    
    bbox_n = np.stack((x0,y0,x1,y1), axis=1)
    
    return bbox_n

def xy2wh(bbox):
    #change format from  x1,y1, x2, y2 -> xc,yc,w h
    xc = (bbox[...,0] + bbox[...,2])/2
    yc = (bbox[...,1] + bbox[...,3])/2
    w = (bbox[...,2] - bbox[...,0])
    h = (bbox[...,3] - bbox[...,1])
    
    
    bbox_n = np.stack((xc,yc,w,h), axis=1)
    
    return bbox_n

def get_jaccard_index(box_a,box_b):
    def get_area(box):
        return (box[:,2]-box[:,0]) * (box[:,3] - box[:,1])
    
    #this is a fast method to find what boxes have some intersection  
    min_xy = np.maximum(box_a[:, None, :2], box_b[None, :, :2])
    max_xy = np.minimum(box_a[:, None, 2:], box_b[None, :, 2:])
    
    #negative value here means no inteserction
    inter = (max_xy-min_xy).clip(min=0)
    inter = inter[...,0] * inter[...,1]
    
    union = get_area(box_a)[:,None] + get_area(box_b)[None] - inter
    
    jaccard = inter/union
    
    return jaccard  

class BoxEncoder():
    def __init__(self, *args, **argkws):
        self._anc_wh, self.n_anchors_shapes = get_anchor_boxes(*args, **argkws)
        self._anc_xy = wh2xy(self._anc_wh )
    
    def encode(self, labels, bboxes):
        
        if labels.size > 0:
        
            #get classification targets
            #boxes that has a jaccard index > 0.5 are positive examples,
            #boxes that has a jaccard index < 0.4 are negative examples,
            #antying in between would ignored
            
            overlaps = get_jaccard_index(bboxes, self._anc_xy)
            
            if overlaps.size == 0:
                import pdb
                pdb.set_trace()
            
            
            #larger overlap with each of the elements in the grid
            max_ids = np.argmax(overlaps, axis=0) 
            max_vals = np.amax(overlaps, axis=0)
            
            clf_target = labels[max_ids]
            is_bgnd = max_vals < 0.4
            is_maybe = (max_vals < 0.5) & ~is_bgnd
            
            clf_target[is_bgnd] = 0
            clf_target[is_maybe] = -1
            
            #get regression targets following ssd losses
            loc_ = xy2wh(bboxes[max_ids, :])
            
            #transform the raw coordinates according to the anchor boxes
            xr = (loc_[:, 0]-self._anc_wh[:, 0])/self._anc_wh[:, 2]
            yr = (loc_[:, 1]-self._anc_wh[:, 1])/self._anc_wh[:, 3]
            wr = np.log(loc_[:, 2]/self._anc_wh [:, 2])
            hr = np.log(loc_[:, 3]/self._anc_wh [:, 3])
            loc_target = np.stack((xr, yr, wr, hr), axis=1)  
        else:
            nn = self._anc_xy.shape[0]
            clf_target, loc_target = np.zeros(nn, np.int), np.zeros((nn,4), np.float32)
        
        
        return clf_target, loc_target
    
    def decode(self, labels, bboxes):
        #%%
        pos_ = labels>0
        out_lab = labels[pos_] 
        
        bb = bboxes[pos_]
       
        x = bb[:, 0]*self._anc_wh[pos_, 2] + self._anc_wh[pos_, 0]
        y = bb[:, 1]*self._anc_wh[pos_, 3] + self._anc_wh[pos_, 1]
        w = np.exp(bb[:, 2])*self._anc_wh[pos_, 2]
        h = np.exp(bb[:, 3])*self._anc_wh[pos_, 3]
        
        out_loc = np.stack((x, y, w, h), axis=1)  
        out_loc = wh2xy(out_loc)
        
        return out_lab, out_loc

if __name__ == '__main__':
    anchors, n_shapes = get_anchor_boxes()
    
    anchors_xy = xy2wh(anchors)
    anchors_wh = wh2xy(anchors_xy)
    
    assert np.max(np.abs((anchors_wh-anchors))) < 1e-5