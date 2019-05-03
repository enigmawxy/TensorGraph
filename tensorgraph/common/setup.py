# setup.py
# export CFLAGS="-I /Users/xiaoyongwang/Robot/tfenv/lib/python3.7/site-packages/numpy/core/include $CFLAGS"
# python setup.py build_ext --inplace
from distutils.core import setup
from Cython.Build import cythonize
# import numpy as np

setup(
    ext_modules=cythonize('cutil.pyx'),
    include_dirs=["/Users/xiaoyongwang/Robot/tfenv/lib/python3.7/site-packages/numpy/core/include"]
)