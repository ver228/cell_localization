#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:51:05 2019

@author: avelinojaver
"""
from ..utils import save_checkpoint

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
    
import tqdm
import time
import torch

__all__ = ['train_mask']
def train_one_epoch(basename, model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    
    model.train()
    header = f'{basename} Train Epoch: [{epoch}]'
    
    avg_loss = 0
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        preds = model(images)
        loss = criterion(preds, targets)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        avg_loss += loss.item()
        
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
    avg_loss /= len(data_loader)
     
    #save data into the logger
    logger.add_scalar('train_loss', avg_loss, epoch)
    
    return avg_loss


@torch.no_grad()
def evaluate_one_epoch(basename, model, criterion, optimizer, lr_scheduler, data_flow, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    model.eval()
    header = f'{basename} Test Epoch: [{epoch}]'
    
    avg_loss = 0
    model_time_avg = 0
    
    
    N = len(data_flow.data_indexes)
    for ind in tqdm.trange(N, desc = header):
        image, target = data_flow.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = image.to(device)
        target = target.to(device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        
        preds = model(image[None])
        
        model_time_avg += time.time() - model_time
        
        loss = criterion(preds, target[None])
    
        avg_loss += loss.item()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
    model_time_avg /= N
    avg_loss /= N
    
    #save data into the logger
    logger.add_scalar('test_loss', avg_loss, epoch)
    logger.add_scalar('model_time', model_time_avg, epoch)
    
    return avg_loss

def train_mask(save_prefix,
        model,
        device,
        criterion,
        train_flow,
        val_flow,
        optimizer,
        log_dir,
        
        lr_scheduler = None,
        
        batch_size = 128,
        n_epochs = 2000,
        num_workers = 4
        ):
    
    
    train_loader = DataLoader(train_flow, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers
                            )
    
    
    
    model = model.to(device)
    
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_score = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
            
        train_one_epoch(save_prefix, 
                         model, 
                         criterion,
                         optimizer, 
                         lr_scheduler, 
                         train_loader, 
                         device, 
                         epoch, 
                         logger
                         )
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        test_loss = evaluate_one_epoch(save_prefix, 
                         model, 
                         criterion,
                         optimizer, 
                         lr_scheduler, 
                         val_flow, 
                         device, 
                         epoch, 
                         logger
                          )
        
        
        
        desc = 'epoch {} , loss = {}'.format(epoch, test_loss)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()
            }
        
        
        is_best = test_loss < best_score
        if is_best:
            best_score = test_loss
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        
    
        