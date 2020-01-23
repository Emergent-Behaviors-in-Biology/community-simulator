#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:40:34 2017

@author: robertmarsland
"""

from setuptools import setup

setup(name='community_simulator',
      version='1.0',
      description='Simulate batch culture experiments on microbial communities.',
      url='https://github.com/Emergent-Behaviors-in-Biology/community-simulator',
      author='Robert Marsland III',
      author_email='robertvsiii@gmail.com',
      license='MIT',
      packages=['community_simulator'],
      install_requires=['cvxpy>=1',
      					'numpy',
      					'pandas',
      					'matplotlib',
      					'scipy'],
      python_requires='>=3',
      zip_safe=False)
