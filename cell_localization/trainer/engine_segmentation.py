#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
#import multiprocessing as mp
#mp.set_start_method('spawn', force=True)

from ..flow import collate_simple
from ..utils import save_checkpoint
from ..evaluation import get_masks_metrics, get_IoU_best_match

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch.utils.tensorboard import SummaryWriter

from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import time
import tqdm
import numpy as np

__all__ = ['train_segmentation']

def mask_metrics2scores(metrics, logger, prefix, epoch):
    scores = {}
    for iclass, (TP, FP, FN,  agg_inter, agg_union) in metrics.items():
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        AJI = agg_inter/agg_union
        
        scores[iclass] = (P, R, F1, AJI)
        
        
        
        logger.add_scalar(f'{prefix}_P_{iclass}', P, epoch)
        logger.add_scalar(f'{prefix}_R_{iclass}', R, epoch)
        logger.add_scalar(f'{prefix}_F1_{iclass}', F1, epoch)
        logger.add_scalar(f'{prefix}_AJI_{iclass}', AJI, epoch)
    
    return scores

def train_one_epoch(basename, model, optimizer, lr_scheduler, data_loader, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    
    model.train()
    header = f'{basename} Train Epoch: [{epoch}]'
    
    train_avg_losses = defaultdict(int)
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    for images, targets in pbar:
        
        images = torch.from_numpy(np.stack(images)).to(device)
        targets = [{k: torch.from_numpy(v).to(device) for k, v in target.items()} for target in targets]
        
        losses = model(images, targets)
        
        
        loss = sum([x for x in losses.values()])
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5) # I was having problems here before. I am not completely sure this makes a difference now
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        for k,l in losses.items():
            train_avg_losses[k] += l.item()
     
        
    train_avg_losses = {k: loss / len(data_loader) for k, loss in train_avg_losses.items()} #average loss
    train_avg_loss = sum([x for x in train_avg_losses.values()]) # total loss
    
    #save data into the logger
    for k, loss in train_avg_losses.items():
        logger.add_scalar('train_' + k, loss, epoch)
    logger.add_scalar('train_epoch_loss', train_avg_loss, epoch)
    
    return train_avg_loss


@torch.no_grad()
def evaluate_one_epoch(basename, model, data_loader, device, epoch, logger, eval_dist = 5):
    model.eval()
    
    header = f'{basename} Test Epoch: [{epoch}]'
    
    metrics = {'all' : np.zeros(5)}
    
    model_time_avg = 0
    test_avg_losses = defaultdict(int)
    
    N = len(data_loader.data_indexes)
    for ind in tqdm.trange(N, desc = header):
        image, target = data_loader.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        
        losses, pred_segmentation = model(image[None], [target])
        
        model_time_avg += time.time() - model_time
        
        for k,l in losses.items():
            test_avg_losses[k] += l.item()
        
        
        true_cells_mask = (target['segmentation_mask']==1).cpu().numpy().astype(np.uint8)
        pred_cells_mask = (pred_segmentation[0] == 1).cpu().numpy().astype(np.uint8)
        
        pred_coords, target_coords, IoU, agg_inter, agg_union = get_masks_metrics(true_cells_mask, pred_cells_mask)
        TP, FP, FN, pred_ind, true_ind = get_IoU_best_match(IoU)
        
        
        metrics['all'] += TP, FP, FN, agg_inter, agg_union
    
    
    model_time_avg /= N
    test_avg_losses = {k:loss/N for k, loss in test_avg_losses.items()} #get the average...
    test_avg_loss = sum([x for x in test_avg_losses.values()]) #... and the total loss
    
    #save data into the logger
    for k, loss in test_avg_losses.items():
        logger.add_scalar('val_' + k, loss, epoch)
    logger.add_scalar('val_avg_loss', test_avg_loss, epoch)
    logger.add_scalar('model_time', model_time_avg, epoch)
    
    scores = mask_metrics2scores(metrics, logger, 'val', epoch)
    
    AJI = scores['all'][-1]
    
    return AJI


def train_segmentation(save_prefix,
        model,
        device,
        train_flow,
        val_flow,
        optimizer,
        log_dir,
        
        lr_scheduler = None,
        
        batch_size = 16,
        n_epochs = 2000,
        num_workers = 1,
        init_model_path = None,
        save_frequency = 200,
        hard_mining_freq = None,
        val_dist = 5
        ):
    
    
    train_loader = DataLoader(train_flow, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers,
                            collate_fn = collate_simple,
                            )

    
    model = model.to(device)
    
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_score = 0#1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        train_one_epoch(save_prefix, 
                         model, 
                         optimizer, 
                         lr_scheduler, 
                         train_loader, 
                         device, 
                         epoch, 
                         logger
                         )
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        val_AJI = evaluate_one_epoch(save_prefix, 
                           model, 
                           val_flow, 
                           device, 
                           epoch, 
                           logger,
                           val_dist
                           )
        
        desc = f'epoch {epoch} , AJI={val_AJI}'
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'model_input_parameters' : model.input_parameters,
                'train_flow_input_parameters': train_loader.dataset.input_parameters,
                'val_flow_input_parameters': train_loader.dataset.input_parameters
            }
        
        is_best = val_AJI > best_score
        if is_best:
            best_score = val_AJI
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
 
    
        