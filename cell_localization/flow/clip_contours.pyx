#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:58:18 2019

@author: avelinojaver
"""
import numpy as np
cimport numpy as np
cimport cython


ctypedef fused my_type:
    int
    float
    double
    long long

# cdef means here that this function is a plain C function (so faster).
# To get all the benefits, we type the arguments and the return value.
cdef my_type clip(my_type a, my_type min_value, my_type max_value):
    return min(max(a, min_value), max_value)

@cython.boundscheck(False)
@cython.wraparound(False)
def c_crop_contour(my_type[:, :] coords,  my_type xl,  my_type xr,  my_type yl,  my_type yr):
    cdef Py_ssize_t tot = coords.shape[0]
    assert coords.shape[1] == 2
    
    if my_type is int:
        dtype = np.intc
    elif my_type is cython.float:
        dtype = np.float32
    elif my_type is cython.double:
        dtype = np.float64
    elif my_type is cython.longlong:
        dtype = np.longlong
    
    coords_clipped = np.zeros([tot, 2], dtype=dtype)
    cdef my_type[:, ::1] result_view = coords_clipped
    
    cdef my_type xlim = xr - xl - 1
    cdef my_type ylim = yr - yl - 1   
 
    cdef my_type x, y
     
    cdef my_type clip_xmin = xlim
    cdef my_type clip_xmax = 0
    cdef my_type clip_ymin = ylim
    cdef my_type clip_ymax = 0
    
    
    
    cdef Py_ssize_t i
    cdef char has_valid_elements = 0
    
    for i in range(tot):
        x = clip(coords[i, 0] - xl, 0, xlim)
        y = clip(coords[i, 1] - yl, 0, ylim)
        
        if not (x == 0 or x == xlim):
            clip_ymin = min(clip_ymin, y)
            clip_ymax = max(clip_ymax, y)
            has_valid_elements = 1
        
        if not (y == 0 or y == ylim):
            clip_xmin = min(clip_xmin, x)
            clip_xmax = max(clip_xmax, x)
            has_valid_elements = 1
        
        result_view[i, 0] = x 
        result_view[i, 1] = y
    
    if not has_valid_elements:
        return np.zeros((0, 2), dtype=dtype)
    
    for i in range(tot):
        result_view[i, 0] = clip(result_view[i, 0], clip_xmin, clip_xmax)
        result_view[i, 1] = clip(result_view[i, 1], clip_ymin, clip_ymax)
        
    return coords_clipped