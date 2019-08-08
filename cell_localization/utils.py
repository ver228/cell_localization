#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:10:33 2019

@author: avelinojaver
"""

def add_input_params(obj):
    def wrapper(*args, **argkws):
        input_params = [args, argkws]
        return obj(*args, **argkws, input_params = input_params)
    return wrapper