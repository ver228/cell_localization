#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from .helper import save_checkpoint

from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


import tqdm
import numpy as np


def train_demixer(save_prefix,
        model,
        device,
        train_flow,
        criterion,
        optimizer,
        log_dir,
        
        batch_size = 16,
        n_epochs = 2000,
        num_workers = 1,
        init_model_path = None,
        save_frequency = 200
        ):
    
    epoch_init = 0
    train_loader = DataLoader(train_flow, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)

    
    model = model.to(device)
    
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_loss = 1e10
    pbar_epoch = tqdm.trange(epoch_init, n_epochs)
    for epoch in pbar_epoch:
        
        #train
        model.train()
        pbar = tqdm.tqdm(train_loader, desc = f'{save_prefix} Train')        
        train_avg_loss = 0
        for X, target in pbar:
            assert not np.isnan(X).any()
            assert not np.isnan(target).any()
            
            X = X.to(device)
            target = target.to(device)
            pred = model(X)
            
            loss = criterion(pred, target)
            
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            
            
            clip_grad_norm_(model.parameters(), 0.5) # I was having problems here if i do not include batch norm...
            
            optimizer.step() 
        
            train_avg_loss += loss.item()
            
            
        
        train_avg_loss /= len(train_loader)
        logger.add_scalar('train_epoch_loss', train_avg_loss, epoch)
        
        
        desc = 'epoch {} , loss={}'.format(epoch, train_avg_loss)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch + epoch_init,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        
        is_best = train_avg_loss < best_loss
        if is_best:
            best_loss = train_avg_loss  
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
 