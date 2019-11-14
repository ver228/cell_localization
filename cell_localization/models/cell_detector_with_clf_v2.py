#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 15:41:15 2019

@author: avelinojaver
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .unet import init_weights
from .cell_detector import BeliveMapsNMS, get_loss

class HeadClassifier(nn.Sequential):
    def __init__(self, 
                 n_inputs, 
                 n_classes, 
                 n_filters, 
                 dropout_p = 0.5
                 ):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        
        self.conv = nn.Sequential(
                nn.Conv2d(n_inputs, n_filters, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                )
        
        self.pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(n_filters, n_classes),
                )
        
        #initialize model
        for m in self.modules():
            init_weights(m, init_type = 'xavier')
        
    def forward(self, xin):
        x = self.conv(xin)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        xout = self.fc(x)
        return xout



    
class ProposalHeadClassifierFC(nn.Sequential):
    def __init__(self, n_inputs, n_classes, dropout_p = 0.2):
        super().__init__()
        
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        
        self.fc_head = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Linear(n_inputs, 1024),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Linear(1024, 1024),
                )
        
        
        self.fc_clf = nn.Sequential(
                nn.Linear(1024, n_classes)
                )
        for m in self.modules():
            init_weights(m, init_type = 'xavier')
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.fc_head(x)
        xout = self.fc_clf(x)
        
        return xout


class CellDetectorWithClassifierInd(nn.Module):
    def __init__(self, 
                 mapping_network,
                 n_classes,
                 loss_type = 'l2-G1.5',
                 
                 nms_threshold_abs = 0.2,
                 nms_threshold_rel = None,
                 nms_min_distance = 3,
                 proposal_size = 11,
                 roi_pool_size = 7,
                 return_belive_maps = False
                 ):
        
        
        super().__init__()
        
        _dum = set(dir(self))
        
        assert mapping_network.n_outputs == 1
        self.mapping_network = mapping_network
        self.n_classes = n_classes
        
        self.nms_threshold_abs = nms_threshold_abs
        self.nms_threshold_rel = nms_threshold_rel
        self.nms_min_distance = nms_min_distance
        
        
        
        self.proposal_size = proposal_size
        self.proposal_half_size = self.proposal_size//2
        self.proposal_n_max = 2000
        self.proposal_match_r2 = int(self.proposal_half_size**2) # I want to avoid having to change the coords to float
        self.roi_pool_size = (roi_pool_size, roi_pool_size)
        
        
        self.loss_type = loss_type
        
        self._input_names = list(set(dir(self)) - _dum) #i want the name of this fields so i can access them if necessary
        
        
        
        self.criterion_patch_loc, self.preevaluation = get_loss(loss_type)
        self.nms = BeliveMapsNMS(nms_threshold_abs, nms_threshold_rel, nms_min_distance)
        
        self.return_belive_maps = return_belive_maps
        
        _n_inputs = self.mapping_network.down_blocks[-1].n_filters[-1]
        self.clf_patch_head = HeadClassifier(n_inputs = _n_inputs, n_classes = 2, n_filters = 256)
        self.criterion_patch_clf = nn.CrossEntropyLoss()
        
        #_n_inputs = self.mapping_network.up_blocks[-1].n_filters[-1]*(roi_pool_size**2)
        _n_inputs = self.mapping_network.up_blocks[-1].n_filters[-1]
        self.clf_proposal_head = HeadClassifier(n_inputs = _n_inputs,n_classes =  n_classes + 1, n_filters = 256)
        self.criterion_proposal_clf = nn.CrossEntropyLoss()
        
        
    @property
    def input_parameters(self):
        return {x:getattr(self, x) for x in self._input_names}
    
    def forward(self, x, targets = None):
        xhat, features = self.mapping_network(x)
        
        
        outputs = []
        if self.training or (targets is not None):
            patch_clf_scores = self.clf_patch_head(features[0])
            
            
            #coordinates to region targets. Here I am expecting to be trained on patches
            valid_ = [t['labels'].shape[0]>0 for t in targets]
            valid_ = torch.tensor(valid_, device = xhat.device)
            valid_loc_xhat = xhat[valid_]
            valid_last_feats = features[-1][valid_]
            assert xhat.shape[-2:] == valid_last_feats.shape[-2:] #otherwise i cannot really calculate the feature boxes
            
            valid_loc_targets = [t for t in targets if t['labels'].shape[0]>0]
            patch_clf_target = valid_.long()
            
            loss_patch_loc = self.criterion_patch_loc(valid_loc_xhat, valid_loc_targets)
            loss_patch_clf = self.criterion_patch_clf(patch_clf_scores, patch_clf_target)
            
            
            valid_loc_xhat = self.preevaluation(valid_loc_xhat)
            outs = self.nms(valid_loc_xhat)
            assert len(valid_loc_targets) == len(outs)
            
            proposals = []
            target_labels = []
            for pred, true in zip(outs, valid_loc_targets):
                pred_coords = pred[0]
                
                #sort values by the relative score
                pred_score = pred[-1]
                inds = torch.argsort(pred_score, descending = True)
                inds = inds[:self.proposal_n_max]
                pred_coords = pred_coords[inds]
                
                offsets_mat = pred_coords.view(-1, 1, 2) - true['coordinates'].view(1, -1, 2)
                dists2 = (offsets_mat**2).sum(dim=-1)
                min_dist2, match_id = dists2.min(dim=1)
                
                
                valid_matches = min_dist2 <= self.proposal_match_r2 # I am using the squre distanc to avoid having to cast to float
                
                n_pred = len(pred_coords)
                labels = torch.zeros(n_pred, dtype = torch.long, device=pred_coords.device, requires_grad = False)
                labels[valid_matches] = true['labels'][match_id[valid_matches]]
                target_labels.append(labels)
                
                
                boxes = torch.cat((pred_coords - self.proposal_half_size, pred_coords + self.proposal_half_size), dim = -1)
                proposals.append(boxes) 
                
                #offsets = torch.zeros((n_pred, 2), dtype = torch.float, device=pred_coords.device, requires_grad = False)
                #offsets_min = offsets_mat[np.arange(offsets_mat.shape[0]), match_id]
                #offsets[valid_matches] = offsets_min[valid_matches].float()/self.proposal_half_size
                #target_offsets.append(offsets)
            
            target_labels = torch.cat(target_labels)
            proposals = [x.float() for x in proposals] # I need to optimize this 
            assert len(proposals) == len(valid_last_feats)
            pooled_feats = roi_align(valid_last_feats, proposals, self.roi_pool_size, 1)
            assert len(target_labels) == len(pooled_feats)
            
            predicted_labels = self.clf_proposal_head(pooled_feats)
            loss_proposal_clf = self.criterion_proposal_clf(predicted_labels, target_labels)
            
            losses = dict(
                    loss_patch_loc = loss_patch_loc*0.01,
                    loss_patch_clf = loss_patch_clf,
                    loss_proposal_clf = loss_proposal_clf,
                    )
            outputs.append(losses)
            
        
        
        if not self.training:
            #I want to get a map to indicate if there is an cell or not
            feats = features[0].permute((0, 2, 3, 1))
            n_batch, clf_h, clf_w, clf_n_filts = feats.shape
            
            #feed each region of the lower feature layer to the patch classifier
            feats = feats.contiguous().view(-1, clf_n_filts, 1, 1)
            clf_patch_scores = self.clf_patch_head(feats)
            
            #get the probability of each pixel, reshape and upscale
            clf_patch_scores = F.softmax(clf_patch_scores, dim = 1)            
            clf_patch_scores = clf_patch_scores[:, 1].view(n_batch, 1, clf_h, clf_w)
            clf_patch_scores = F.interpolate(clf_patch_scores, size = xhat.shape[-2:], mode = 'bilinear', align_corners=False)
            
            
            bad = clf_patch_scores< 0.5
            xhat[bad] = xhat[bad].mean()
            xhat = self.preevaluation(xhat)
            outs = self.nms(xhat)
            
            proposals = []
            for pred in outs:
                pred_coords = pred[0]
                boxes = torch.cat((pred_coords - self.proposal_half_size, pred_coords + self.proposal_half_size), dim = -1)
                proposals.append(boxes) 
            
            
            proposals = [x.float() for x in proposals] # I need to optimize this 
            pooled_feats = roi_align(features[-1], proposals, self.roi_pool_size, 1)
            
            labels_v = self.clf_proposal_head(pooled_feats)
            
            #I need to reshape to match the original number per each element in the batch
            proposal_predictions = []
            for x in proposals:
                n = len(x)
                proposal_predictions.append((labels_v[:n]))
                labels_v = labels_v[n:]
                
            assert labels_v.numel() == 0
            
    
            predictions = []
            for preds, out in zip(proposal_predictions, outs):
                
                preds_labels = preds
                scores_labels, labels = preds_labels.max(dim = -1)
                
                coordinates, _, scores_abs, scores_rel = out
                
                _valid = labels > 0
                res = dict(
                            coordinates = coordinates[_valid],
                            labels = labels[_valid],
                            scores_labels = scores_labels[_valid],
                            scores_abs = scores_abs[_valid],
                            scores_rel = scores_rel[_valid],
                            
                            )

                predictions.append(res)
            
            outputs.append(predictions)

        if self.return_belive_maps:
            if self.training:
                outputs.append(xhat)
            else:
                outputs.append((xhat, clf_patch_scores))
        
        if len(outputs) == 1:
            outputs = outputs[0]
        
        return outputs


            