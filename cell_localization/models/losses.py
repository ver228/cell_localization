#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:27:31 2018

@author: avelinojaver
"""

from torch import nn
import torch.nn.functional as F
import torch
import math

class L0AnnelingLoss(nn.Module):
    def __init__(self, anneling_rate=1/50):
        super().__init__()
        
        self.anneling_rate = anneling_rate
        self._n_calls = 0
        self._init_gamma = 2
        self._last_gamma = 0
        self._eps = 1e-8
    
    def forward(self, input_v, target):
        gamma = max(self._init_gamma - self._n_calls*self.anneling_rate, self._last_gamma)
        self._n_calls += 1
        
        return ((input_v-target).abs() + self._eps).pow(gamma).sum()



class BootstrapedPixL2(nn.Module):
    '''bootstrapped pixel-wise L2 loss'''
    def __init__(self, bootstrap_factor=4):
        super().__init__()
        self.bootstrap_factor = bootstrap_factor
        
    def forward(self, input_v, target):
        mat_l2 = torch.pow(input_v-target,2)
        mat_l2 = mat_l2.view(mat_l2.shape[0],-1)
        out, _ = torch.topk(mat_l2, 4, dim=1)
        return out.sum()


class AdjustSmoothL1Loss(nn.Module):
    #modified from https://github.com/chengyangfu/retinamask/blob/master/maskrcnn_benchmark/layers/adjust_smooth_l1_loss.py
    #RetinaMask: Learning to predict masks improves state-of-the-art single-shot detection for free
    def __init__(self, momentum =  0.1, beta = 1.0):
        super(AdjustSmoothL1Loss, self).__init__()
        self.momentum = momentum
        self.beta_hat = beta
        
        #i am only minimizing one parameter so i just need one beta
        self.running_mean = beta
        self.running_var = 0
        
    def forward(self, inputs, target, size_average=True):

        n = torch.abs(inputs -target)
        with torch.no_grad():
            if torch.isnan(n.var(dim=0)).sum().item() == 0:
                m = n.mean().item()
                s = n.var().item()
                
                self.running_mean = (1 - self.momentum)*self.running_mean + (self.momentum * m)
                self.running_var = (1 - self.momentum)*self.running_var + (self.momentum * s)
                

        beta = (self.running_mean - self.running_var)
        
        #clamp beta [0, beta_hat]
        beta = max(beta, 1e-3)
        beta = min(beta, self.beta_hat)
        loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        
        if size_average:
            return loss.mean()
        else:
            return loss.sum()
    
class FocalLoss(nn.Module):
    def __init__(self, 
                 num_classes, 
                 alpha = 0.25, 
                 gamma = 2.):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        
        self.hinge_loss = nn.SmoothL1Loss(reduction='sum')
        
    
    def focal_loss(self, preds, targets):
        target_onehot = torch.eye(self.num_classes+1)[targets]
        target_onehot = target_onehot[:,1:].contiguous() #the zero is the background class
        target_onehot = target_onehot.to(targets.device) #send to gpu if necessary
        
        focal_weights = self._get_weight(preds,target_onehot)
        
        #I already applied the sigmoid to the classification layer. I do not need binary_cross_entropy_with_logits
        return (focal_weights*F.binary_cross_entropy(preds, target_onehot, reduce=False)).sum()
    
    def _get_weight(self, x, t):
        pt = x*t + (1-x)*(1-t)
        w = self.alpha*t + (1-self.alpha)*(1-t)
        return w * (1-pt).pow(self.gamma)
    
    def forward(self, pred, target):
        clf_target, loc_target = target
        clf_preds, loc_preds = pred
        
        ### regression loss
        pos = clf_target > 0
        num_pos = pos.sum().item()
        
        #since_average true is equal to divide by the number of possitives
        loc_loss = self.hinge_loss(loc_preds[pos], loc_target[pos])
        loc_loss = loc_loss/max(1, num_pos)
        
        #### focal lost
        valid = clf_target >= 0  # exclude ambigous anchors (jaccard >0.4 & <0.5) labelled as -1
        clf_loss = self.focal_loss(clf_preds[valid], clf_target[valid])
        clf_loss = clf_loss/max(1, num_pos)  #inplace operations are not permitted for gradients
        
        
        #I am returning both losses because I want to plot them separately
        return clf_loss, loc_loss


class MaskFocalLoss(nn.Module):
    def __init__(self,  
                 alpha = 0.25, 
                 gamma = 2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        self.hinge_loss = nn.SmoothL1Loss(reduction='sum')
        
    
    def focal_loss(self, pred, target):
        pred_s = torch.sigmoid(pred)
        target_s = torch.sigmoid(target)
        
        
        
        focal_weights = self._get_weight(pred_s,target_s)
        
        #I already applied the sigmoid to the classification layer. I do not need binary_cross_entropy_with_logits
        return (focal_weights*F.binary_cross_entropy(pred_s, target_s, reduce=False)).sum()
    
    def _get_weight(self, x, t):
        pt = x*t + (1-x)*(1-t)
        w = self.alpha*t + (1-self.alpha)*(1-t)
        return w * (1-pt).pow(self.gamma)
    
    def forward(self, pred, target):
        pos = target > 0.1
        num_pos = pos.sum().item()
        
        #since_average true is equal to divide by the number of positives
        loc_loss = self.hinge_loss(pred[pos], target[pos])
        loc_loss = loc_loss/max(1, num_pos)
        
        #### focal lost
        clf_loss = self.focal_loss(pred, target)
        clf_loss = clf_loss/max(1, num_pos)  #inplace operations are not permitted for gradients
        
        loss = clf_loss + loc_loss
        
        return loss

class MaximumLikelihoodLoss(nn.Module):
    #based on https://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18a/neumann18a.pdf
    def __init__(self, poolbychannel = False):
        super().__init__()
        self.poolbychannel = poolbychannel
        
    def forward(self, pred, target):
        n_batch, n_out, w, h = pred.shape
        pred = pred.contiguous()
        target = target.contiguous()
        
        #I am assuming the target is zeros and ones, where ones are the coordinates to predict
        if self.poolbychannel:
            target_l = target.transpose(0, 1).contiguous().view(n_out, -1)
            pred_l = pred.transpose(0, 1).contiguous().view(n_out, -1)
            pred_l = nn.functional.log_softmax(pred_l, dim= 1)
            
            
            #ch_losses = -(pred_l*target_l).sum(dim = 1)/target_l.sum(dim = 1)
            #loss = ch_losses.sum()
            
            loss = -(pred_l*target_l).sum()/target_l.sum()
        else:
            target_l = target.view(n_batch, n_out, -1)
            pred_l = pred.view(n_batch, n_out, -1)
            pred_l = nn.functional.log_softmax(pred_l, dim= 2)
            loss = -(pred_l*target_l).sum()/target_l.sum()
            
        
        return loss

class MixtureModelLoss(nn.Module):
    #based on https://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18a/neumann18a.pdf
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        #I am assuming the output channels a multiple of 5 so it can contain all the model parameters 
        n_batch, n_out, w, h = pred.shape
        n_labs = n_out // 5
        
        target_l = target.transpose(0, 1).contiguous().view(n_labs, -1)
        pred_l = pred.transpose(0, 1).contiguous().view(n_labs, 5, -1)
        
        out_s = pred_l[:, 0]
        
        out_s = nn.functional.log_softmax(out_s, dim= 1)
        
        
        mux = torch.clamp(pred_l[:, 1], -3, 3)
        muy = torch.clamp(pred_l[:, 2], -3, 3)
        sx = torch.clamp(pred_l[:, 3], 1, 100)
        sy = torch.clamp(pred_l[:, 4], 1, 100)
    
        eps = 1e-8
        #I am assuming a rho of zero (no correlation between x and y in the bivariate gaussian)
        # 1/(2pi*sy*sx)*exp(-(dx^2/sx^2 + dy^2/sy^2)/2)
        
        z1 = -((mux*mux)/(sx*sx + eps) + (muy*muy)/(sy*sy + eps))/2 
        z2 = -torch.log((2*math.pi*sx*sy + 1.)) #i add one for numerical stability 
        
        p = target_l*(out_s + z1 + z2)
        if torch.isnan(p).any() or torch.isinf(z1).any() or torch.isinf(z2).any():
            raise
        
        loss = -p.sum(dim=1)/(target_l.sum(dim=1) + eps)
        loss = torch.mean(loss)
        
        if loss < 0:
            import pdb
            pdb.set_trace()
        
        return loss
  


def get_loss(loss_type):
    if loss_type == 'l1':
        criterion = nn.L1Loss()
    
    elif loss_type == 'l1smooth':
        criterion = nn.SmoothL1Loss()
    
    elif loss_type == 'adjl1smooth':
        criterion = AdjustSmoothL1Loss()
    
    elif loss_type == 'l2':
        criterion = nn.MSELoss()
    
    elif loss_type == 'maxlikelihood':
        criterion = MaximumLikelihoodLoss()
        
    elif loss_type == 'maxlikelihoodpooled':
        criterion = MaximumLikelihoodLoss(poolbychannel = True)
    
    elif loss_type == 'mixturemodelloss':
        criterion = MixtureModelLoss()
    
    elif loss_type == 'l0anneling':
        criterion = L0AnnelingLoss(anneling_rate=1/50)
    
    elif loss_type == 'bootpixl2':
        criterion = BootstrapedPixL2(bootstrap_factor=4)
        
    elif loss_type == 'maskfocal':
        criterion = MaskFocalLoss()

    elif loss_type.startswith('focal-'):
        #should be `focal-{num_classes}`
        num_classes = int(loss_type.split('-')[1])
        criterion = FocalLoss(num_classes)
        
    else:
        raise ValueError(loss_type)
    return criterion