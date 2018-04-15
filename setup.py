from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('\'numpy\' package is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('\'scipy\' package is required during installation')
    sys.exit(1)

setup(name='COSFIRE',
      version='0.0.3',
      description='COSFIRE machine learning and image processing',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      )
