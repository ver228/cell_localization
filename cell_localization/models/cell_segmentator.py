#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:41:15 2019

@author: avelinojaver
"""
import torch
from torch import nn

class LossFromLabels(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
        
    def _prepare(self, pred, label):
        
        batch_size, num_classes, height, width = pred.shape
        target_one_hot = torch.zeros(batch_size, 
                              num_classes, 
                              height, 
                              width,
                              device = pred.device, 
                              dtype = pred.dtype)
        
        target_one_hot.scatter_(1, label.unsqueeze(1), 1.0) + self.eps
        
        pred_prob = torch.softmax(pred, dim=1)
        
        return pred_prob, target_one_hot
        


class DiceLoss(LossFromLabels):
    # based on: https://github.com/kornia/kornia/blob/master/kornia/losses/dice.py
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_prob, target_prob = self._prepare(pred, target)
        
        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(pred_prob * target_prob, dims)
        cardinality = torch.sum(pred_prob + target_prob, dims)

        dice_score = 2. * (intersection + self.eps) / (cardinality + self.eps)
        
        return torch.mean(torch.tensor(1.) - dice_score)
    

class GeneralizedDiceLoss(LossFromLabels):
    def __init__(self):
        #generilized dice loss https://arxiv.org/pdf/1707.03237.pdf 
        super().__init__()

    def forward(self, pred, target):
        pred_prob, target_prob = self._prepare(pred, target)
        
        #from https://arxiv.org/pdf/1707.03237.pdf 
        # "weighting, the contribution of each label is corrected by the inverse of its volume, thus reducing the well known correlation between region size and Dice score"
        
        #[batch, n_classes]
        counts = target_prob.sum(dim = (2,3))
        
        counts[counts<1] = 1
        #counts = torch.zeros(3)
        weights = 1/(counts**2)
        
        
        #[batch, n_classes]
        intersection = torch.sum(pred_prob * target_prob,  dim = (2, 3))
        cardinality = torch.sum(pred_prob + target_prob, dim = (2, 3))
        
        weighted_intersection = torch.sum(weights*intersection)
        weighted_cardinality = torch.sum(weights*cardinality)
        
        
        dice_score = 2. * (weighted_intersection + self.eps) / (weighted_cardinality + self.eps)
        
        return 1 - dice_score
    
class WeightedCrossEntropy(LossFromLabels):
    def __init__(self):
        #weighted cross entropy https://arxiv.org/pdf/1707.03237.pdf originally used in https://arxiv.org/abs/1505.04597
        super().__init__()

    def forward(self, pred, target):
        pred_prob, target_prob = self._prepare(pred, target)
        
        #tot pixels
        N = pred_prob.shape[-1]*pred_prob.shape[-2]
        #foreground counts
        counts = pred_prob.sum(dim = (2,3)) + self.eps
        weights = (N-counts)/counts
        
        eps_logit = 1e-6 #if this value is too low torch.log will throw `inf`
        pred_prob = torch.clamp(pred_prob, eps_logit, 1 - eps_logit)
        logits = weights[..., None, None]*target_prob*torch.log(pred_prob) + (1-target_prob)*torch.log(1-pred_prob)
        
        return -torch.mean(logits)

class TverskyLoss(nn.Module):
    # based on: https://github.com/kornia/kornia/blob/master/kornia/losses/dice.py
    def __init__(self, alpha, beta):
        #https://arxiv.org/pdf/1706.05721.pdf 
        #"by adjusting the hyperparameters α and β we can control the trade-off between false positives and false negatives"
        #alpha = beta = 0.5 -> dice coefficient
        #alpha = beta = 0.5 ->  Tanimoto coefficient, similar to Jaccard?
        #alpha + beta = 1 -> F_{beta} scores
        super().__init__()
        
    def forward(self, pred, target):
        pred_prob, target_prob = self._prepare(pred, target)
        
        
        dims = (1, 2, 3)
        intersection = torch.sum(pred_prob * target_prob, dims)
        fps = torch.sum(pred_prob * (torch.tensor(1.) - target_prob), dims)
        fns = torch.sum((torch.tensor(1.) - pred_prob) * target_prob, dims)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = (numerator + self.eps) / (denominator + self.eps)
        return torch.mean(torch.tensor(1.) - tversky_loss)

def get_loss(loss_type):
    
    if loss_type.startswith('crossentropy'):
        weight = None
        if '+W' in loss_type:
            #'crossentropy+W1-1-10'
            weight = list(map(float, loss_type.partition('+W')[-1].split('-')))
            weight = torch.tensor(weight)
        
        criterion = nn.CrossEntropyLoss(weight = weight)
        
    elif loss_type == 'WCE':
        criterion = WeightedCrossEntropy()
        
    elif loss_type == 'dice':
        criterion = DiceLoss()
    
    elif loss_type == 'GDL':
        criterion = GeneralizedDiceLoss()
        
        
    elif loss_type.startswith('tversky'):
        #I am expecting something like tversky-0.3-07
        _, alpha, beta = loss_type.split('-')
        criterion = TverskyLoss(float(alpha), float(beta))
        
        
    else:
        raise ValueError('Not implemented {loss_type}')
        
    return criterion

class CellSegmentator(nn.Module):
    def __init__(self, 
                 mapping_network,
                 
                 loss_type = 'crossentropy+W1-1-10',
                 return_belive_maps = False
                 ):
        
        
        super().__init__()
        
        _dum = set(dir(self))
        
        self.n_classes = mapping_network.n_outputs
        self.mapping_network_name = mapping_network.__name__
        self.loss_type = loss_type
        self.return_belive_maps = return_belive_maps
        
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        self.mapping_network = mapping_network
        
        
        self.criterion_segmentation = get_loss(self.loss_type)
    
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
        
    
    def forward(self, x, targets = None):
        xhat = self.mapping_network(x)
        
        
        outputs = []
        
        if self.training or (targets is not None):
            x_target = torch.stack([t['segmentation_mask'] for t in targets])
            loss = dict(
                loss_clf = self.criterion_segmentation(xhat, x_target)
                )
            
            outputs.append(loss)
        
        
        if not self.training:
            labels = xhat.argmax(dim=1)
            outputs.append(labels)
            pass

        if self.return_belive_maps:
            outputs.append(xhat)
            
        
        if len(outputs) == 1:
            outputs = outputs[0]
        
        return outputs

if __name__ == '__main__':
    
    mapping_network = mapping_network = lambda x : x
    mapping_network.n_outputs = 3
    #%%
    x = torch.rand((1, 3, 512, 512))
    targets = [{'segmentation_mask':torch.randint(0, 3, (512, 512))}]
    model = CellSegmentator(mapping_network, return_belive_maps= True, loss_type = 'WCE')
    losses, xhat = model(x, targets)
    

    #%%
    