#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 14:15:42 2018

@author: avelinojaver
"""
import math
import torch
from torch import nn
from torchvision.models.detection.rpn import AnchorGenerator, concat_box_prediction_layers, det_utils
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from collections import OrderedDict

def _norm_init_weights(m): 
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
        nn.init.constant_(m.bias.data, 0.0)
        

class BasicHead(nn.Sequential):
    def __init__(self, in_planes, num_anchors):
        layers = []
        for i in range(4):
            _n_in = in_planes if i == 0 else 256
            conv = nn.Conv2d(_n_in, 256, kernel_size=3, stride=1, padding=1)
            layers.append(conv)
            layers.append(nn.ReLU())
        super().__init__(*layers)
        
        for m in self.modules():
            _norm_init_weights(m)


class ClassificationHead(nn.Module):
    def __init__(self, in_planes, num_classes, num_anchors, is_classification = False):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self._head = BasicHead(in_planes, num_anchors)
        
        conv = nn.Conv2d(256, num_classes*num_anchors, kernel_size=3, stride=1, padding=1)
        #add bias to make easier to train the classification layer 
        #"every anchor should be labeled as foreground with confidence of ∼π"
        probability = 0.01
        bias = -math.log((1-probability)/probability)
        nn.init.constant_(conv.bias.data, bias)
        nn.init.constant_(conv.weight.data, 0.0)
        
        self.clf = nn.Sequential(conv, 
                                 nn.Sigmoid()
                                 )
        
    def forward(self, x):
        x = self._head(x)
        pred = self.clf(x)
        
        #batch_size = pred.shape[0]
        #pred = pred.permute(0,2,3,1).contiguous().view(batch_size, -1, self.num_classes)
        
        return pred
    
class RegressionHead(nn.Module):
    def __init__(self, in_planes, num_anchors, is_classification = False):
        super().__init__()
        self.num_anchors = num_anchors
        
        self._head = BasicHead(in_planes, num_anchors)
        self.loc = nn.Conv2d(256, 4*num_anchors, kernel_size=3, stride=1, padding=1)
        _norm_init_weights(self.loc)
        
    def forward(self, x):
        x = self._head(x)
        pred = self.loc(x)
        
        #batch_size = pred.shape[0]
        #pred = pred.permute(0,2,3,1).contiguous().view(batch_size, -1, 4)
        
        return pred

class RetinaHead(nn.Module):
    def __init__(self, in_planes, num_classes, num_anchors):
        super().__init__()
        
        self.in_planes = in_planes
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        self.bbox_pred = RegressionHead(self.in_planes, self.num_anchors)
        self.cls_logits = ClassificationHead(self.in_planes, self.num_classes, self.num_anchors)
        
    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            logits.append(self.cls_logits(feature))
            bbox_reg.append(self.bbox_pred(feature))
        return logits, bbox_reg

#%%
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha = 0.25, gamma = 2.):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
    
    def focal_loss(self, preds, targets):
        target_onehot = torch.eye(self.num_classes+1)[targets]
        target_onehot = target_onehot[:,1:].contiguous() #the zero is the background class
        target_onehot = target_onehot.to(targets.device) #send to gpu if necessary
        
        focal_weights = self._get_weight(preds,target_onehot)
        
        #I already applied the sigmoid to the classification layer. I do not need binary_cross_entropy_with_logits
        return (focal_weights*nn.F.binary_cross_entropy(preds, target_onehot, reduce=False)).sum()
    
    def _get_weight(self, x, t):
        pt = x*t + (1-x)*(1-t)
        w = self.alpha*t + (1-self.alpha)*(1-t)
        return w * (1-pt).pow(self.gamma)
    
    def forward(self, pred, target):
        #%%
        clf_target, loc_target = target
        clf_preds, loc_preds = pred
        
        ### regression loss
        pos = clf_target > 0
        num_pos = pos.sum().item()
        
        #since_average true is equal to divide by the number of possitives
        loc_loss = nn.F.smooth_l1_loss(loc_preds[pos], loc_target[pos], size_average=False)
        loc_loss = loc_loss/max(1, num_pos)
        
        #### focal lost
        valid = clf_target >= 0  # exclude ambigous anchors (jaccard >0.4 & <0.5) labelled as -1
        clf_loss = self.focal_loss(clf_preds[valid], clf_target[valid])
        clf_loss = clf_loss/max(1, num_pos)  #inplace operations are not permitted for gradients
        
        
        #I am returning both losses because I want to plot them separately
        return clf_loss, loc_loss

#%%
class RetinaNet(nn.Module):
    def __init__(self, 
                 backbone, 
                 num_classes = 1, 
                 anchor_generator=None, 
                 head=None,
                 nms_thresh=0.5,
                 image_mean = [1., 1., 1.], 
                 image_std= [0., 0., 0.],
                 min_size = 512,
                 max_size = 512
                 ):
        
        super().__init__()
        
        
        
        if anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )    
            
        out_channels = backbone.out_channels
        if head is None:
            head = RetinaHead(
                out_channels, num_classes, anchor_generator.num_anchors_per_location()[0]
            )
        
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        self.box_coder = det_utils.BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        self.anchor_generator = anchor_generator
        self.head = head
        self.backbone = backbone
        self.focal_loss = FocalLoss(num_classes)
        
    def forward(self, images, targets=None):
#        """
#        Arguments:
#            images (list[Tensor]): images to be processed
#            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
#
#        Returns:
#            result (list[BoxList] or dict[Tensor]): the output from the model.
#                During training, it returns a dict[Tensor] which contains the losses.
#                During testing, it returns list[BoxList] contains additional fields
#                like `scores`, `labels` and `mask` (for Mask R-CNN models).
#
#        """
        #if self.training and targets is None:
        #    raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        
        features = list(features.values())
        clf_scores, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level = [o[0].numel() for o in clf_scores]
        clf_scores, pred_bbox_deltas = \
            concat_box_prediction_layers(clf_scores, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        
        #detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        
#
#        losses = {}
#        losses.update(detector_losses)
#        losses.update(proposal_losses)
#
#        if self.training:
#            return losses
#
        return clf_scores, pred_bbox_deltas

if __name__ == '__main__':
    backbone = resnet_fpn_backbone(backbone_name = 'resnet50', pretrained = False)
    model = RetinaNet(backbone = backbone, min_size = 128, max_size = 128)
    
    #%%
    from flow import BBBC042Dataset, collate_simple
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    data_dir = Path('/Users/avelinojaver/Downloads/BBBC042/')
    gen = BBBC042Dataset(data_dir, max_samples = 10, roi_size = 256)
    loader = DataLoader(gen, batch_size = 2, collate_fn = collate_simple)