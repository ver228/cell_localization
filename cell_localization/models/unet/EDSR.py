#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:46:30 2019

@author: avelinojaver
#based on https://github.com/thstkdgus35/EDSR-PyTorch/

"""


from torch import nn

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, 
                     out_channels, 
                     kernel_size,
                     padding=(kernel_size//2), 
                     bias=bias)


class BasicBlock(nn.Sequential):
    def __init__(self, 
                 conv, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 bias=False, 
                 bn=True, 
                 act=nn.ReLU(inplace = True)
                 ):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)
        
        
class ResBlock(nn.Module):
    def __init__(self, 
                 conv, 
                 n_feats, 
                 kernel_size,
                 bias=True, 
                 bn=False, 
                 act=nn.ReLU(inplace = True), 
                 res_scale=1
                 ):
        
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    
    

#    16, 64, 1.
#    32, 256, 0.1
class EDSRBody(nn.Module):
    def __init__(self, 
                 n_inputs = 1,
                 n_outputs = 1,
                 n_resblocks = 16, 
                 n_feats = 64,
                 res_scale = 1.
                 ):
        super(EDSRBody, self).__init__()
        
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        kernel_size = 3 
        
        act = nn.ReLU(True)
        
        # define head module
        m_head = [default_conv(n_inputs, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                default_conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(default_conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [default_conv(n_feats, n_outputs, kernel_size)]
        
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        
        return x 