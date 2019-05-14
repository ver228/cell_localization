#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 23:29:46 2018

@author: avelinojaver
"""
import tqdm
import datetime
import os
import shutil
import torch
import numpy as np

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader 

from .flow import VesicleBBFlow, root_dir, collate_fn
from .models import RetinaNet, FocalLoss

log_dir_root = root_dir / 'results' / 'localization'

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_path)

def train(
    num_classes = 2,
    model_name = 'retinanet',
    cuda_id = 0,
    batch_size = 16,
    loss_type = 'adam',
    lr = 1e-5,
    n_epochs = 500,
    backbone = 'resnet34',
    is_clean_data = False
    ):
    if torch.cuda.is_available():
        print("THIS IS CUDA!!!!")
        dev_str = "cuda:" + str(cuda_id)
    else:
        dev_str = 'cpu'
    device = torch.device(dev_str)
    
    gen = VesicleBBFlow(is_clean_data=is_clean_data)
    loader = DataLoader(gen, 
                        batch_size = batch_size, 
                        shuffle = True, 
                        collate_fn = collate_fn)
    
    
    model = RetinaNet(num_classes=num_classes, 
                      num_anchors = gen.encoder.n_anchors_shapes,
                      backbone = backbone)
    criterion = FocalLoss(num_classes)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model = model.to(device)
    
    now = datetime.datetime.now()
    bn = now.strftime('%Y%m%d_%H%M%S') + '_' + model_name + '-' + backbone
    if is_clean_data:
        bn += '_clean'
    
    bn = '{}_{}_{}_lr{}_batch{}'.format(loss_type, bn, 'adam', lr, batch_size)
    log_dir = log_dir_root / bn
    logger = SummaryWriter(log_dir = str(log_dir))
    
    #%%
    best_loss = 1e10
    pbar_epoch = tqdm.trange(n_epochs)
    for epoch in pbar_epoch:
        
        #train
        model.train()
        model.freeze_bn()
        
        gen.train()
        pbar = tqdm.tqdm(loader)
        
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
        
        avg_losses /= len(loader)
        tb = [('train_loss', avg_losses[0]),
              ('train_loss_clf', avg_losses[1]),
              ('train_loss_loc', avg_losses[2])
            ]
        
        for tt, val in tb:
            logger.add_scalar(tt, val, epoch)
        
        
        #test
        model.eval()
        gen.test()
        
        avg_losses = np.zeros(3)
        
        with torch.no_grad():
            pbar = tqdm.tqdm(loader)
            for X, target in pbar:
                X = X.to(device)
                target = [x.to(device) for x in target]
                pred = model(X)
                
                clf_loss, loc_loss = criterion(pred, target)
                loss = clf_loss + loc_loss
                
                for il, ll in enumerate((loss, clf_loss, loc_loss)):
                    avg_losses[il] += ll.item()
                
                
        avg_losses /= len(loader)
        
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