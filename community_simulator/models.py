#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:03:48 2017

@author: robertmarsland
"""

import numpy as np

def dNdt_CRM(N,R,params):
    c,m,Kinv,r = params
    return N*(np.dot(c,R)-m)

def dRdt_CRM(N,R,params):
    c,m,Kinv,r = params
    return r*R*(1-Kinv*R)-R*np.dot(c.T,N)