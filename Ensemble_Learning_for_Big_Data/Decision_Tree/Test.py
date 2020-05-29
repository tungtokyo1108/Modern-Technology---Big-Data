#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:04:14 2020

@author: tungbioinfo
"""

import numpy as np
import matplotlib.pyplot as plt
import cos_doubles

x = np.arange(0, 2 * np.pi, 0.1)
y = np.empty_like(x)

cos_doubles.cos_doubles_func(x, y)
plt.plot(x,y)

###############################################################################

import _utils

_utils.test(10)

_utils.log(10)

import Test_tree

Test_tree.log_func(100)



























