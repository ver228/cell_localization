#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 23:29:46 2018

@author: avelinojaver
"""
from .helper import save_checkpoint

import tqdm
import torch
import numpy as np

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader 


def train_bboxes(save_prefix,
        model,
        device,
        train_flow,
        val_flow,
        criterion,
        optimizer,
        log_dir,
        
        batch_size = 16,
        n_epochs = 2000,
        num_workers = 1,
        init_model_path = None,
        save_frequency = 200
        ):
    
    model = model.to(device)

    train_loader = DataLoader(train_flow, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)
    val_loader = DataLoader(val_flow, 
                            batch_size=batch_size,
                            num_workers=num_workers)
    
    
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        
        #train
        model.train()
        model.freeze_bn()
        pbar = tqdm.tqdm(train_loader, desc = f'{save_prefix} Train :')
        
        avg_losses = np.zeros(3)
        
        for X, target in pbar:
            X = X.to(device)
            target = [x.to(device) for x in target]
            pred = model(X)
            
            clf_loss, loc_loss = criterion(pred, target)
            loss = clf_loss + loc_loss
            
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step() 
            
            for il, ll in enumerate((loss, clf_loss, loc_loss)):
                avg_losses[il] += ll.item()
        
        avg_losses /= len(train_loader)
        tb = [('train_loss', avg_losses[0]),
              ('train_loss_clf', avg_losses[1]),
              ('train_loss_loc', avg_losses[2])
            ]
        
        for tt, val in tb:
            logger.add_scalar(tt, val, epoch)
        
        
        #test
        model.eval()
        
        avg_losses = np.zeros(3)
        
        with torch.no_grad():
            pbar = tqdm.tqdm(val_loader, desc = f'{save_prefix} Test :')
            for X, target in pbar:
                X = X.to(device)
                target = [x.to(device) for x in target]
                pred = model(X)
                
                clf_loss, loc_loss = criterion(pred, target)
                loss = clf_loss + loc_loss
                
                for il, ll in enumerate((loss, clf_loss, loc_loss)):
                    avg_losses[il] += ll.item()
                
                
        avg_losses /= len(val_loader)
        
        tb = [('test_loss', avg_losses[0]),
              ('test_loss_clf', avg_losses[1]),
              ('test_loss_loc', avg_losses[2])
            ]
        for tt, val in tb:
            logger.add_scalar(tt, val, epoch)
        
        desc = 'epoch {} , loss={}'.format(epoch, avg_losses[0])
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        is_best = avg_losses[0] < best_loss
        save_checkpoint(state, is_best, save_dir = str(log_dir))