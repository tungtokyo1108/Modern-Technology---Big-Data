#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:42:19 2020

@author: tungbioinfo
"""

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy 


setup(
      cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("_utils", 
                   sources=["_utils.pyx"],
                   include_dirs=[numpy.get_include()]),
                   Extension("Test_tree",
                   sources=["Test_tree.pyx"],
                   include_dirs=[numpy.get_include()]),
                   Extension("_tree",
                   sources=["_tree.pyx"],
                   include_dirs=[numpy.get_include()]),
                   Extension("_criterion",
                   sources=["_criterion.pyx"],
                   include_dirs=[numpy.get_include()]),
                   Extension("_splitter",
                   sources=["_splitter.pyx"],
                   include_dirs=[numpy.get_include()]),
                   Extension("_Decision_Tree",
                   sources=["Decision_Tree.pyx"],
                   include_dirs=[numpy.get_include()])],
      )






