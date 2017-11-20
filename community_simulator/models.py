#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:03:48 2017

@author: robertmarsland
"""

import numpy as np

def dNdt_CRM(N,R,params):
    return N*(np.dot(params['c'],(R*params['w']))-params['m'])

def dRdt_CRM(N,R,params):
    return params['r']*R*(1-params['Kinv']*R)-R*np.dot(params['c'].T,N)