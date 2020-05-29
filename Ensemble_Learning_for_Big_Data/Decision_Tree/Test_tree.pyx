#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:18:01 2020

@author: tungbioinfo
"""

cimport _utils
import _utils
import numpy as np
cimport numpy as np

def main():
    
    cdef double a = 10
    print("Log is", _utils.log(a))

def log_func(double x):
    return _utils.log(x)

