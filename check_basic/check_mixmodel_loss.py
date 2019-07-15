#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:19:28 2019

@author: avelinojaver
"""

import torch
from torch import nn
import tqdm
import numpy as np
import matplotlib.pylab as plt
import math
import cv2
#simple proof of concept of this paper: https://www.robots.ox.ac.uk/~vgg/publications/2018/Neumann18a/neumann18a.pdf

#%%

n_batches = 2
n_segs = 3

coords = [(0, 1, 8, 8), (1, 1, 8, 24), (0, 0, 24, 24), (0, 0, 24, 8), (1, 2, 16, 16)]

X = np.zeros((n_batches, n_segs, 32, 32))
for cc in coords:
    X[cc[0], cc[1], cc[2], cc[3]] = 1.


for bb in range(n_batches):
    for ii in range(n_segs):
        X[bb, ii] = cv2.GaussianBlur(X[bb, ii], (3,3), 1)
X /= X.max()
X = torch.from_numpy(X).float()



target = torch.zeros((n_batches, n_segs, 32, 32), requires_grad = False)
for cc in coords:
    target[cc[0], cc[1], cc[2], cc[3]] = 1.


net = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(32, 8, 3, padding = 1),
        nn.ReLU(inplace = True),
        nn.Conv2d(8, 5*n_segs, 3, padding = 1),
        nn.ReLU(inplace = True)
        )

for m in net.parameters():
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.xavier_normal_(m.weight.data)

lr = 1e-2
optimizer = torch.optim.Adam(net.parameters(), lr = lr)

losses = []
for ii in tqdm.trange(10000):
    pred = net(X)
    n_batch, n_out, w, h = pred.shape
    n_segs = n_out // 5
    
    target_l = target.transpose(0, 1).contiguous().view(n_segs, -1)
    pred_l = pred.transpose(0, 1).contiguous().view(n_segs, 5, -1)
    
    out_s = pred_l[:, 0]
    
    out_s = nn.functional.log_softmax(out_s, dim= 1)
    
    
    mux = torch.clamp(pred_l[:, 1], -3, 3)#pred_l[:, 1]#
    muy = torch.clamp(pred_l[:, 2], -3, 3)#pred_l[:, 2]#
    sx = torch.clamp(pred_l[:, 3], 1, 100)
    sy = torch.clamp(pred_l[:, 4], 1, 100)
    
    eps = 1e-8
    #I am assuming a rho of zero (no correlation between x and y in the bivariate gaussian)
    # 1/(2pi*sy*sx)*exp(-(dx^2/sx^2 + dy^2/sy^2)/2)
    
    z1 = -((mux*mux)/(sx*sx + eps) + (muy*muy)/(sy*sy + eps))/2 
    z2 = -torch.log((2*math.pi*sx*sy + eps)) 
    
    p = target_l*(out_s + z1 + z2)
    if torch.isnan(p).any() or torch.isinf(z1).any() or torch.isinf(z2).any():
        raise
    
    loss = -p.sum(dim=1)/(target_l.sum(dim=1) + eps)
    loss = torch.mean(loss)
    
    optimizer.zero_grad()   
    
    
    loss.backward()                     # backpropagation, compute gradients
    losses.append(loss.item())
    optimizer.step()

#%%
out_t = target[0].view(n_segs, 32,32).detach().numpy()
out_t = np.rollaxis(out_t, 0, 3)


out_r = out_s.reshape(n_segs, -1, w, h).exp()[:, 0].detach().numpy()
out_r /= out_r.max()

dx = mux.reshape(n_segs, -1, w, h)[:, 0].detach().numpy()
dy = muy.reshape(n_segs, -1, w, h)[:, 0].detach().numpy()
ssy = sy.reshape(n_segs, -1, w, h)[:, 0].detach().numpy()
ssx = sx.reshape(n_segs, -1, w, h)[:, 0].detach().numpy()

out_r, dx, dy, ssx, ssy = [np.rollaxis(x, 0, 3)  for x in [out_r, dx, dy, ssx, ssy]]


fig, axs = plt.subplots(1, 2, sharex= True, sharey=True)

axs[0].imshow(out_t)
axs[0].set_title('True')


axs[1].imshow(out_r)
axs[1].set_title('Prediction')

fig, axs = plt.subplots(2, 2, sharex= True, sharey=True)

axs[0][0].imshow(dx)
axs[0][1].imshow(dy)
axs[1][0].imshow(ssx)
axs[1][1].imshow(ssy)
