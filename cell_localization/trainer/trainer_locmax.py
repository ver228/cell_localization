#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:46:42 2018

@author: avelinojaver
"""
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from .helper import save_checkpoint
from cell_localization.evaluation.localmaxima import evaluate_coordinates
from cell_localization.flow.flow_coords import coords2mask

from tensorboardX import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


import tqdm
import numpy as np
from skimage.feature import peak_local_max


def train_locmax(save_prefix,
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
        save_frequency = 200,
        hard_mining_freq = None
        ):
    
    epoch_init = 0
    
    train_loader = DataLoader(train_flow, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers)
#    val_loader = DataLoader(val_flow, 
#                            batch_size=batch_size,
#                            num_workers=num_workers
#                            )
    
    peaks_local_args = dict(
            min_distance = 3, 
            threshold_abs = 0.05, 
            threshold_rel = 0.1
            )
    
#    peaks_local_args = dict(
#            min_distance = 3, 
#            threshold_abs = 0.1, 
#            threshold_rel = 0.5
#            )
    
    model = model.to(device)
    
    
    log_dir = log_dir / save_prefix
    logger = SummaryWriter(log_dir = str(log_dir))
    
    
    best_loss = 1e10
    pbar_epoch = tqdm.trange(epoch_init, n_epochs)
    for epoch in pbar_epoch:
        #Hard Mining
        if hard_mining_freq is not None and epoch > 0 and (epoch + 1) % hard_mining_freq == 0:
            
            
            #initialize structure
            hard_neg_data = {}
            for _type, _type_data in train_flow.slides_data.items():
                hard_neg_data[_type] = {}
                for _slide, _slide_data in _type_data.items():
                    hard_neg_data[_type][_slide] = [None]*len(_slide_data)
                
            
            model.eval()
            metrics = np.full((model.n_classes, 3), 1e-3)
            with torch.no_grad():
                N = len(train_flow.slide_data_indexes)
                
                for ind in tqdm.trange(N, desc = 'Hard Mining'):
                    _type, _slide, _img_id = train_flow.slide_data_indexes[ind]
                    
                    xin, coords_rec = train_flow.read_full(ind)
                
                    xin = torch.from_numpy(xin[None])
                    xin = xin.to(device)
                    xhat = model(xin)
                
                    xout = xhat[0].detach().cpu().numpy()
                    
                    bads = []
                    for iclass in range(model.n_classes):
                        _class = iclass + 1
                        rec = coords_rec[coords_rec['type_id'] == _class]
                        target = np.array((rec['cy'], rec['cx'])).T
                        preds = peak_local_max(xout[iclass], **peaks_local_args)
                        TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(preds, target, max_dist = 5)
                        metrics[iclass] += (TP, FP, FN)
                        
                        if target.size > 0:
                            good = np.zeros(preds.shape[0], np.bool)
                            good[pred_ind] = True
                            pred_bad = preds[~good]
                        else:
                            pred_bad = preds
                        
#                        if preds.size > 0:
#                            good = np.zeros(target.shape[0], np.bool)
#                            good[true_ind] = True
#                            target_bad_rec = rec[~good]
#                        else:
#                            target_bad_rec = rec
                        
                        cols = ['type_id', 'cx', 'cy', 'radius']
                        r_dtypes = [(col, coords_rec.dtype[col]) for col in cols]
                        r_data = [(-_class, x, y, train_flow.min_radius)for y,x in pred_bad]
                        pred_bad_rec = np.array(r_data, r_dtypes)
                        
                        bads.append(pred_bad_rec)
                        #bads += [pred_bad_rec, target_bad_rec]
                    
                    bads = np.concatenate(bads)
                    hard_neg_data[_type][_slide][_img_id] = bads
                
                train_flow.hard_neg_data = hard_neg_data
                for ind, met in enumerate(metrics):
                    TP, FP, FN = met
                    P = TP/(TP+FP)
                    R = TP/(TP+FN)
                    F1 = 2*P*R/(P+R)
                    
                    logger.add_scalar(f'train_class{ind}_P', P, epoch) 
                    logger.add_scalar(f'train_class{ind}_R', R, epoch) 
                    logger.add_scalar(f'train_class{ind}_F1', F1, epoch)
                
            
                    
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
        
        
        model.eval()
        test_avg_loss = 0
        
        metrics = np.full((model.n_classes, 3), 1e-3)
        with torch.no_grad():
            N = len(val_flow.slide_data_indexes)
            for ind in tqdm.trange(N, desc = f'{save_prefix} Test'):
                img, coords_rec = val_flow.read_full(ind)
            
                xin = torch.from_numpy(img[None])
                xin = xin.to(device)
                mask_preds = model(xin)
            
                xout = mask_preds[0].detach().cpu().numpy()
                
                mask_target = []
                for iclass in range(model.n_classes):
                    _class = iclass + 1
                    rec = coords_rec[coords_rec['type_id'] == _class]
                    target = np.array((rec['cy'], rec['cx'])).T
                    
                    preds = peak_local_max(xout[iclass], **peaks_local_args)
                    TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(preds, target, max_dist = 5)
                    metrics[iclass] += (TP, FP, FN)
                    
                    xys = np.array((rec['cx'], rec['cy']))
                    mm = coords2mask(xys, img.shape[1:], sigma = val_flow.loc_gauss_sigma)
                    mask_target.append(mm)
                    
                mask_target = np.array(mask_target)
                mask_target = torch.from_numpy(mask_target[None])
                mask_target = mask_target.to(device)
                
                loss = criterion(mask_preds, mask_target)
                test_avg_loss += loss.item()
                
            test_avg_loss /= N
                
#        with torch.no_grad():
#            metrics = np.full((model.n_classes, 3), 1e-3)
#            for X, target in pbar:
#                assert not np.isnan(X).any()
#                assert not np.isnan(target).any()
#                
#                X = X.to(device)
#                target = target.to(device)
#                pred = model(X)
#                
#                loss = criterion(pred, target)
#                test_avg_loss += loss.item()
#                
#                for p, t in zip(pred, target):
#                    for ind in range(model.n_classes):
#                        p_map = p[ind].detach().cpu().numpy()
#                        p_coords = peak_local_max(p_map, min_distance = 3, threshold_abs = 0.1, threshold_rel = 0.5)
#                        
#                        
#                        t_map = t[ind].detach().cpu().numpy()
#                        t_coords = peak_local_max(t_map, min_distance = 3, threshold_abs = 0.1, threshold_rel = 0.5)
#                    
#                        TP, FP, FN, pred_ind, true_ind = evaluate_coordinates(p_coords, t_coords, max_dist = 5)
#                        metrics[ind] += (TP, FP, FN)
#                    
#                
#        
        
        
        logger.add_scalar('test_epoch_loss', test_avg_loss, epoch)
        
        for ind, met in enumerate(metrics):
            TP, FP, FN = met
            P = TP/(TP+FP)
            R = TP/(TP+FN)
            F1 = 2*P*R/(P+R)
            
            logger.add_scalar(f'test_class{ind}_P', P, epoch) 
            logger.add_scalar(f'test_class{ind}_R', R, epoch) 
            logger.add_scalar(f'test_class{ind}_F1', F1, epoch)
            
        avg_loss = -F1#test_avg_loss
        
        desc = 'epoch {} , loss={}'.format(epoch, avg_loss)
        pbar_epoch.set_description(desc = desc, refresh=False)
        
        state = {
                'epoch': epoch + epoch_init,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
        
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss  
        save_checkpoint(state, is_best, save_dir = str(log_dir))
        
        if (epoch+1) % save_frequency == 0:
            checkpoint_path = log_dir / f'checkpoint-{epoch}.pth.tar'
            torch.save(state, checkpoint_path)
 