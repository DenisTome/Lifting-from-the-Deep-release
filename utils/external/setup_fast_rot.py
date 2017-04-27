# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:15:33 2016

@author: chrisr
"""
try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = 'Fast Rotation estimate',
  ext_modules=[
    Extension('upright_fast',
              sources=['upright_fast.pyx'],
              extra_compile_args=["-Ofast", "-mfpmath=sse", "-msse4.2" ,"-ffast-math", "-funroll-loops",
                                  "-march=native", "-fomit-frame-pointer"],
              language='c++',
              include_dirs=[numpy.get_include()])
    ],
  cmdclass = {'build_ext': build_ext}
)
