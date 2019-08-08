#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:34:06 2019

@author: avelinojaver
"""
import os
import torch
import shutil

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)

def get_device(cuda_id):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    return device

def get_optimizer(optimizer_name, model, lr, weight_decay = 0.0, momentum = 0.9):
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model_params, lr = lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model_params, lr = lr, momentum = momentum, weight_decay = weight_decay)
    else:
        raise ValueError('Invalid optimizer name {}'.format(optimizer_name))
        
    return optimizer