#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:27:31 2018

@author: avelinojaver
"""

from torch import nn
import torch.nn.functional as F
import torch

class LossWithBeliveMaps(nn.Module):
    def __init__(self, target_loss = F.mse_loss,  gauss_sigma = 2., increase_factor = 1., is_regularized = False, device = None):
        super().__init__()
        
        assert isinstance(gauss_sigma, float)
        assert isinstance(is_regularized, bool)
        
        self.target_loss = target_loss
        self.increase_factor = increase_factor
        self.is_regularized = is_regularized
        
        gaussian_kernel = self.get_gaussian_kernel(gauss_sigma)
        self.gaussian_kernel = nn.Parameter(gaussian_kernel)
    
    @staticmethod
    def get_gaussian_kernel(gauss_sigma, device = None):
        #convolve with a gaussian filter
        kernel_size = int(gauss_sigma*4)
        if kernel_size % 2 == 0:
            kernel_size += 1
        rr = kernel_size//2
    
        x = torch.linspace(-rr, rr, kernel_size, device = device)
        y = torch.linspace(-rr, rr, kernel_size, device = device)
        xv, yv = torch.meshgrid([x, y])
        
        # I am not normalizing it since i do want the center to be equal to 1
        gaussian_kernel = torch.exp(-(xv.pow(2) + yv.pow(2))/(2*gauss_sigma**2))
        return gaussian_kernel
    
   
    def regularize_by_maxima(self, predictions, targets):
        keypoints = []
        for pred, target in zip(predictions, targets):
            coordinates = target['coordinates']
            
            if pred.shape[0] == 1:
                ch_ind = 0
            else:
                #If the channel dimension is more than one this would mean that each channel correspond to a different label
                labels = target['labels']
                assert (labels > 0).all()
                ch_ind = labels - 1
            
            p = pred[ch_ind, coordinates[:, 1], coordinates[:, 0]]
            keypoints.append(p)
            
        keypoints = torch.cat(keypoints)
        keypoints = keypoints.clamp(1e-3, 0.49)
        reg = -keypoints.mean().log()
        return reg
        
    
    def targets2belivemaps(self, targets, expected_shape, device = None):
        masks = torch.zeros(expected_shape, device = device)
        
        
        
        for i, target in enumerate(targets):
            coordinates = target['coordinates']
            
            if expected_shape[1] == 1:
                ch_ind = 0
            else:
                #If the channel dimension is more than one this would mean that each channel correspond to a different label
                labels = target['labels']
                assert (labels > 0).all()
                ch_ind = labels - 1
            
            masks[i, ch_ind, coordinates[:, 1], coordinates[:, 0]] = 1.
            
        kernel_size = self.gaussian_kernel.shape[0]
        gauss_kernel = self.gaussian_kernel.expand(1, expected_shape[1], kernel_size, kernel_size)
        belive_map = F.conv2d(masks, gauss_kernel, padding = kernel_size//2)
        belive_map *= self.increase_factor
        
        return belive_map 
    
    def forward(self, prediction, target):
        target_map = self.targets2belivemaps(target, prediction.shape, device = prediction.device)
        loss = self.target_loss(prediction, target_map)
        
        if self.is_regularized:
            loss += self.regularize_by_maxima(prediction, target)
        
        return loss

class MaximumLikelihoodLoss(nn.Module):
    #based on https://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18a/neumann18a.pdf
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        n_batch, n_out, w, h = predictions.shape
        predictions = predictions.contiguous()
        
        #apply the log_softmax along all the images in a given channel in order to include empty images
        pred_l = predictions.transpose(0, 1).contiguous().view(n_out, -1)
        pred_l = nn.functional.log_softmax(pred_l, dim= 1)
        pred_l = pred_l.view(n_out, n_batch, w, h).transpose(0, 1)
        
        
        keypoints = []
        for pred, target in zip(pred_l, targets):
            coordinates = target['coordinates']
            
            if pred.shape[0] == 1:
                ch_ind = 0
            else:
                #If the channel dimension is more than one this would mean that each channel correspond to a different label
                labels = target['labels']
                assert (labels > 0).all()
                ch_ind = labels - 1
            
            p = pred[ch_ind, coordinates[:, 1], coordinates[:, 0]]
            keypoints.append(p)
        keypoints = torch.cat(keypoints)
        
        loss = -keypoints.mean()
        
        return loss
