#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:15:38 2019

@author: avelinojaver
"""


from extract_eggs import load_model, get_device

from pathlib import Path
from collections import defaultdict
import tqdm
import torch
import numpy as np
import pandas as pd
import cv2

from torch.utils.data import Dataset, DataLoader

def collate_simple(batch):
    return tuple(map(list, zip(*batch)))



class ImagesFlow(Dataset):
    def __init__(self, root_dir, save_dir):
        root_dir = Path(root_dir)
        save_dir = Path(save_dir)
        
        fnames = [x for x in root_dir.rglob('*.png') if not x.name.startswith('.')]
        fnames_with_keys = []
        for fname in fnames:
            base, _, frame = fname.stem.partition('_frame-')
            frame = int(frame)
            
            dname = Path(str(fname.parent).replace(str(root_dir), str(save_dir)))
            save_name = dname / (base + '_eggs-preds.csv')
            
            fnames_with_keys.append((save_name, frame, fname))
        
        self.root_dir = root_dir
        self.fnames_with_keys = fnames_with_keys
    
    def __getitem__(self, ind):
        save_name, frame, fname = self.fnames_with_keys[ind]
        img = cv2.imread(str(fname), -1)
        if img is None:
            raise ValueError(fname)
        
        xin = img[None]
        xin = xin.astype(np.float32)/255
    
        return xin, save_name, frame
    
    def __len__(self):
        return len(self.fnames_with_keys)
    
    

if __name__ == '__main__':
    cuda_id = 0
    
    #bn = 'worm-eggs-adam+Feggsonly+roi96+hard-neg-5_unet-simple_maxlikelihood_20190717_224214_adam_lr0.000128_wd0.0_batch128'
    #nms_threshold_rel = 0.2
    
    bn = 'worm-eggs-adam+Feggs+roi128+hard-neg-5_clf+unet-simple_maxlikelihood_20190803_225943_adam_lr0.000128_wd0.0_batch64'
    nms_threshold_rel = 0.25
    
    
    
    model_path =  Path().home() / 'workspace/localization/results/locmax_detection/eggs/worm-eggs-adam/'/ bn / 'model_best.pth.tar'
    
    #where the masked files are located
    root_dir = Path.home() / 'workspace/WormData/Adam_eggs/'
    
    #where the predictions are going to be stored
    save_dir = Path.home() / 'workspace/localization/predictions/worm_eggs/' / 'Adam_eggs' / bn
    
    
    gen = ImagesFlow(root_dir, save_dir)
    loader = DataLoader(gen, batch_size = 1, num_workers = 4, collate_fn = collate_simple)
    
    
    model_args = dict(nms_threshold_abs = 0.,
                            nms_threshold_rel = nms_threshold_rel,
                            unet_pad_mode = 'reflect')
    
    device = get_device(cuda_id)
    model, epoch = load_model(model_path,**model_args)
    model = model.to(device)
    
    
    
    cuda_id = 0
    device = get_device(cuda_id)
    
    
    data2save = defaultdict(list)
    for X, save_names, frames in tqdm.tqdm(loader):

        with torch.no_grad():
            X = torch.from_numpy(np.stack(X))
            X = X.to(device)
            predictions = model(X)
        
        for save_name, frame_number, prediction in zip(save_names, frames, predictions):
            res = [prediction[x].detach().cpu().numpy() for x in ['coordinates', 'scores_abs', 'scores_rel']]
            res = [x[:, None] if x.ndim == 1 else x for x in res]
            res = np.concatenate(res, axis=1)
            data2save[save_name] += [('/full_data', frame_number, *cc) for cc in zip(*res.T)]
    
    for save_name, data in data2save.items():
         preds_df = pd.DataFrame(data, columns = ['group_name', 'frame_number', 'x', 'y', 'score_abs', 'score_rel'])
         save_name.parent.mkdir(exist_ok=True, parents=True)
         preds_df.to_csv(save_name, index = False)
         

        