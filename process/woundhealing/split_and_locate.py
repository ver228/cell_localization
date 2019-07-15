#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:12:10 2018

@author: avelinojaver
"""
import sys
from pathlib import Path

dname = Path(__file__).resolve().parents[2]
sys.path.append(str(dname))


from cell_localization.models import UNet, UNetv2
from cell_localization.trainer import get_device

import cv2
import torch
import numpy as np
import pandas as pd
import tables
import tqdm

from skimage.feature import peak_local_max

filters = tables.Filters(complevel=0, 
                          complib='blosc', 
                          shuffle=True, 
                          bitshuffle=True, 
                          fletcher32=True
                          )

if __name__ == '__main__':
    is_plot = False
    cuda_id = 0
    #scale_int = (0, 4095)
    img_src_dir = Path.home() / 'workspace/localization/data/woundhealing/manually_filtered/nuclei_and_membrane'
    save_dir_root = Path.home() / 'workspace/localization/data/woundhealing/demixed_predictions'
    
    
    bn = 'cell-demixer-scaled-roi128_unet_l1smooth_20190530_222152_adam_lr0.00032_wd0.0_batch64'
    model_demixer_path = Path.home() / 'workspace/localization/results/cell_demixer' / bn / 'checkpoint.pth.tar'
    assert model_demixer_path.exists()
    
    model_loc_dir = Path.home() / 'workspace/localization/results/locmax_detection/woundhealing/'
    bn = 'woundhealing-no-membrane-roi48_unetv2-init-normal_l1smooth_20190531_140046_adam_lr0.000128_wd0.0_batch128'
    model_loc_path = model_loc_dir / bn.partition('_')[0] / bn / 'model_best.pth.tar'
    assert model_loc_path.exists()
    
    
    device = get_device(cuda_id)
    
    n_ch_in, n_ch_out  = 1, 1
    model_loc = UNetv2(n_channels = n_ch_in, 
                 n_classes = n_ch_out, 
                 batchnorm = False,
                 init_type = None)
    
    
    state = torch.load(model_loc_path, map_location = 'cpu')
    model_loc.load_state_dict(state['state_dict'])
    model_loc.eval()
    model_loc = model_loc.to(device)
    
    n_ch_in, n_ch_out  = 1, 2
    model_demixer = UNet(n_channels = n_ch_in, n_classes = n_ch_out, pad_mode = 'reflect')
    
    
    state = torch.load(model_demixer_path, map_location = 'cpu')
    model_demixer.load_state_dict(state['state_dict'])
    model_demixer.eval()
    model_demixer = model_demixer.to(device)
    #%%
    img_paths = img_src_dir.rglob('*.tif')
    img_paths = list(img_paths)
    for img_path in tqdm.tqdm(img_paths):
        img = cv2.imread(str(img_path), -1)
        
        x = img.astype(np.float32)
        #x = (x - scale_int[0])/(scale_int[1] - scale_int[0])
        img_bot, img_top = x.min(), x.max()
        x = (x - img_bot)/(img_top - img_bot)
        
        with torch.no_grad():
            X = torch.from_numpy(x[None, None])
            X = X.to(device)
            
            X_demixed = model_demixer(X)
            X_nuclei = X_demixed[:, 0]
            X_membrane = X_demixed[:, 1]
            
            Xhat = model_loc(X_nuclei.unsqueeze(1))
            
            x_nuclei = X_nuclei.squeeze().cpu().detach().numpy()
            x_membrane = X_membrane.squeeze().cpu().detach().numpy()
            pred_loc_map = Xhat.squeeze().cpu().detach().numpy()
            
        
        coords_pred = peak_local_max(pred_loc_map, min_distance = 3, threshold_abs = 0.1, threshold_rel = 0.1)
        coords_pred = coords_pred[:, ::-1]
        
        #%%
        save_dir_full = save_dir_root / 'predicted' 
        save_dir_full.mkdir(parents = True, exist_ok = True)
        
        save_dir_rois = save_dir_root / 'rois' 
        save_dir_rois.mkdir(parents = True, exist_ok = True)
        
        #%%
        df = pd.DataFrame({'type_id':1, 
                            'cx':coords_pred[:, 0], 
                            'cy':coords_pred[:, 1]})
        coords_pred_rec = df.to_records(index=False)        

        #%%
        for prefix, img in [('N+M', x), ('M', x_membrane)]: #('N', x_nuclei), 
            img = (img_top - img_bot)*img + img_bot
            img = img.astype(np.uint16)
            
            
            img_id = f'{prefix}_{img_path.stem}'
            save_name = save_dir_full / (img_id + '.hdf5')
            with tables.File(str(save_name), 'w') as fid:
                fid.create_carray('/', 'img', obj = img, filters  = filters)
                fid.create_table('/', 'coords', obj = coords_pred_rec)
            
            # save rois for evaluation
            roi_size = 256
            corner_x = 200
            corner_y = 1
            xr, xl = corner_x, corner_x + roi_size
            yr, yl = corner_y, corner_y + roi_size
            
            
            roi1 = img[xr:xl, yr:yl]
            roi2 = img[xr:xl, -yl:-yr]
            
            
            for roi, sub_str in [(roi1, 'POS'),(roi2, 'NEG')]:
                roi_l = np.log(roi.astype(np.float32))
                bot, top = np.min(roi_l), np.max(roi_l)
                roi_n = ((roi_l - bot)/(top-bot)*255).astype(np.uint8)
                
                save_name = save_dir_rois / f'{img_id}_ROI-LOG-{sub_str}.png'
                save_name.parent.mkdir(exist_ok = True, parents = True)
                cv2.imwrite(str(save_name), roi_n)
                
            
            
        if is_plot:
            import matplotlib.pylab as plt
            fig, axs = plt.subplots(3,1, sharex=True, sharey=True, figsize=(10, 30))
            axs[0].imshow(x, cmap='gray')
            axs[1].imshow(x_nuclei, cmap='gray')
            axs[2].imshow(x_membrane, cmap='gray')
            
            axs[0].plot(coords_pred[..., 0], coords_pred[..., 1], '.r')
            #%%
        