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
from cell_localization.evaluation.localmaxima import score_coordinates

from collections import defaultdict
from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import time
import tqdm
import numpy as np

__all__ = ['train_locmax']
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

def metrics2scores(metrics, logger, prefix, epoch):
    scores = {}
    for iclass, (TP, FP, FN) in metrics.items():
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        scores[iclass] = (P, R, F1)
        
        logger.add_scalar(f'{prefix}_P_{iclass}', P, epoch)
        logger.add_scalar(f'{prefix}_R_{iclass}', R, epoch)
        logger.add_scalar(f'{prefix}_F1_{iclass}', F1, epoch)
    
    return scores

@torch.no_grad()
def evaluate_one_epoch(basename, model, data_loader, device, epoch, logger, eval_dist = 5):
    model.eval()
    
    cpu_device = torch.device("cpu")
    header = f'{basename} Test Epoch: [{epoch}]'
    
    metrics = {'all' : np.zeros(3)}
    for ii in range(1, model.n_classes + 1):
        metrics[ii] = np.zeros(3)
    
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
        
        losses, outputs = model(image[None], [target])
        
        pred = {k: v.to(cpu_device) for k, v in outputs[0].items()}
        
        model_time_avg += time.time() - model_time
        
        for k,l in losses.items():
            test_avg_losses[k] += l.item()
        
        pred_coords = pred['coordinates'].detach().cpu().numpy()
        pred_labels = pred['labels'].detach().cpu().numpy()
            
        target_coords = target['coordinates'].detach().cpu().numpy()
        target_labels = target['labels'].detach().cpu().numpy()
        
        
        TP, FP, FN, pred_ind, true_ind = score_coordinates(pred_coords, target_coords, max_dist = eval_dist)
        metrics['all'] += TP, FP, FN
        
        for iclass in range(1, model.n_classes + 1):
            pred = pred_coords[pred_labels == iclass]
            true = target_coords[target_labels == iclass]
        
            TP, FP, FN, pred_ind, true_ind = score_coordinates(pred, true, max_dist = eval_dist)
            metrics[iclass] += TP, FP, FN
    
    model_time_avg /= N
    test_avg_losses = {k:loss/N for k, loss in test_avg_losses.items()} #get the average...
    test_avg_loss = sum([x for x in test_avg_losses.values()]) #... and the total loss
    
    #save data into the logger
    for k, loss in test_avg_losses.items():
        logger.add_scalar('val_' + k, loss, epoch)
    logger.add_scalar('val_avg_loss', test_avg_loss, epoch)
    logger.add_scalar('model_time', model_time_avg, epoch)
    
    scores = metrics2scores(metrics, logger, 'val', epoch)
    
    F1 = scores['all'][-1]
    
    return F1

@torch.no_grad()
def hard_negative_minig(basename, model, data_loader, device, epoch, logger, eval_dist = 5):
    model.eval()
    
    cpu_device = torch.device("cpu")
    header = f'{basename} Hard  Negative Mining: [{epoch}]'
    
    #initialize structure
    hard_neg_data = {}
    for _type, _type_data in data_loader.data.items():
        hard_neg_data[_type] = {}
        for _group, _group_data in _type_data.items():
            hard_neg_data[_type][_group] = [None]*len(_group_data)
        
    
    metrics = {'all' : np.zeros(3)}
    for ii in range(1, model.n_classes + 1):
        metrics[ii] = np.zeros(3)
    
    N = len(data_loader.data_indexes)
    for ind in tqdm.trange(N, desc = header):
        image, target = data_loader.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        
        outputs = model(image[None])
        pred = {k: v.to(cpu_device) for k, v in outputs[0].items()}
    
        pred_coords = pred['coordinates'].detach().cpu().numpy()
        pred_labels = pred['labels'].detach().cpu().numpy()
            
        target_coords = target['coordinates'].detach().cpu().numpy()
        target_labels = target['labels'].detach().cpu().numpy()
        
        bads_in_image = {'coordinates' : [], 'labels' : []}
        
        TP, FP, FN, pred_ind, true_ind = score_coordinates(pred_coords, target_coords, max_dist = eval_dist)
        metrics['all'] += TP, FP, FN
        
        for iclass in range(1, model.n_classes + 1):
            pred = pred_coords[pred_labels == iclass]
            true = target_coords[target_labels == iclass]
        
            TP, FP, FN, pred_ind, true_ind = score_coordinates(pred, true, max_dist = eval_dist)
            metrics[iclass] += TP, FP, FN
        
            if true.size > 0:
                good = np.zeros(pred.shape[0], np.bool)
                good[pred_ind] = True
                coords_bad = pred[~good]
            else:
                coords_bad = pred
            
            bads_in_image['coordinates'].append(coords_bad)
            lab = np.full(len(coords_bad), -(iclass + 1))
            bads_in_image['labels'].append(lab)
        
        bads_in_image = {k : np.concatenate(v) for k,v in bads_in_image.items()}
        (_type, _group, _img_id) = data_loader.data_indexes[ind]
        hard_neg_data[_type][_group][_img_id] = bads_in_image
        
    data_loader.hard_neg_data = hard_neg_data
    metrics2scores(metrics, logger, 'train', epoch)
    
def train_locmax(save_prefix,
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
    
    
    best_F1 = -1
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        #Hard Mining
        if hard_mining_freq is not None and epoch > 0 and (epoch + 1) % hard_mining_freq == 0:
            hard_negative_minig(save_prefix, model, train_flow, device, epoch, logger, val_dist)
            
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
        
        F1_score = evaluate_one_epoch(save_prefix, 
                           model, 
                           val_flow, 
                           device, 
                           epoch, 
                           logger,
                           val_dist
                           )
        
        
        
        desc = 'epoch {} , F1={}'.format(epoch, F1_score)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'model_input_parameters' : model.input_parameters,
                'train_flow_input_parameters': train_loader.dataset.input_parameters,
                'val_flow_input_parameters': train_loader.dataset.input_parameters
            }
        
        
        is_best = F1_score > best_F1
        if is_best:
            best_F1 = F1_score  
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
 
    
        