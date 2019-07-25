#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
#import multiprocessing as mp
#mp.set_start_method('spawn', force=True)

from ..flow import collate_simple
from .misc import save_checkpoint
from cell_localization.evaluation.localmaxima import score_coordinates

from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import time
import tqdm
import numpy as np
        

def train_one_epoch(basename, model, optimizer, lr_scheduler, data_loader, device, epoch, logger):
     # Modified from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    
    model.train()
    header = f'{basename} Train Epoch: [{epoch}]'
    
    train_avg_loss = 0
    
    pbar = tqdm.tqdm(data_loader, desc = header)
    for images, targets in pbar:
        
        images = torch.from_numpy(np.stack(images)).to(device)
        targets = [{k: torch.from_numpy(v).to(device) for k, v in target.items()} for target in targets]
        
        loss = model(images, targets)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5) # I was having problems here if i do not include batch norm...
        optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        train_avg_loss += loss.item()
        
    train_avg_loss /= len(data_loader)
    logger.add_scalar('train_epoch_loss', train_avg_loss, epoch)
    
    return train_avg_loss    

def metrics2scores(metrics, logger, prefix, epoch):
    
    scores = []
    for ii, (TP, FP, FN) in enumerate(metrics):
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        F1 = 2*P*R/(P+R)
        scores.append((P, R, F1))
        iclass = ii + 1
        logger.add_scalar(f'{prefix}_P_{iclass}', P, epoch)
        logger.add_scalar(f'{prefix}_R_{iclass}', R, epoch)
        logger.add_scalar(f'{prefix}_F1_{iclass}', F1, epoch)
    return scores

@torch.no_grad()
def evaluate_one_epoch(basename, model, data_loader, device, epoch, logger):
    model.eval()
    
    cpu_device = torch.device("cpu")
    header = f'{basename} Test Epoch: [{epoch}]'
    
    metrics = np.zeros((model.n_classes, 3))
    model_time_avg = 0
    test_avg_loss = 0
    
    N = len(data_loader.data_indexes)
    for ind in tqdm.trange(N, desc = header):
        image, target = data_loader.read_full(ind) #I am evaluating one image at the time because some images can have seperated size
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items()}
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        loss, outputs = model(image[None], [target])
        pred = {k: v.to(cpu_device) for k, v in outputs[0].items()}
        
        model_time_avg += time.time() - model_time
        test_avg_loss += loss.item()
        
        pred_coords = pred['coordinates'].detach().cpu().numpy()
        pred_labels = pred['labels'].detach().cpu().numpy()
            
        target_coords = target['coordinates'].detach().cpu().numpy()
        target_labels = target['labels'].detach().cpu().numpy()
        
        for iclass in range(model.n_classes):
            pred = pred_coords[pred_labels == iclass + 1]
            true = target_coords[target_labels == iclass + 1]
        
            
            TP, FP, FN, pred_ind, true_ind = score_coordinates(pred, true, max_dist = 5)
            metrics[iclass, :] += TP, FP, FN
    
    model_time_avg /= N
    test_avg_loss /= N
    logger.add_scalar('model_time', model_time_avg, epoch)
    logger.add_scalar('val_avg_loss', test_avg_loss, epoch)
    
    scores = metrics2scores(metrics, logger, 'val', epoch)
    
    F1_avg = np.mean([x[-1] for x in scores])
    
    return F1_avg

@torch.no_grad()
def hard_negative_minig(basename, model, data_loader, device, epoch, logger):
    model.eval()
    
    cpu_device = torch.device("cpu")
    header = f'{basename} Hard  Negative Mining: [{epoch}]'
    
    #initialize structure
    hard_neg_data = {}
    for _type, _type_data in data_loader.data.items():
        hard_neg_data[_type] = {}
        for _group, _group_data in _type_data.items():
            hard_neg_data[_type][_group] = [None]*len(_group_data)
        
    
    metrics = np.zeros((model.n_classes, 3))
    
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
        for iclass in range(model.n_classes):
            pred = pred_coords[pred_labels == iclass + 1]
            true = target_coords[target_labels == iclass + 1]
        
        
            TP, FP, FN, pred_ind, true_ind = score_coordinates(pred, true, max_dist = 5)
            metrics[iclass, :] += TP, FP, FN
            
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
            hard_negative_minig(save_prefix, model, train_flow, device, epoch, logger)
            
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
        
        F1_avg = evaluate_one_epoch(save_prefix, 
                           model, 
                           val_flow, 
                           device, 
                           epoch, 
                           logger
                           )
        
        
        
        desc = 'epoch {} , F1={}'.format(epoch, F1_avg)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        
        is_best = F1_avg > best_F1
        if is_best:
            best_F1 = F1_avg  
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
 
    
        