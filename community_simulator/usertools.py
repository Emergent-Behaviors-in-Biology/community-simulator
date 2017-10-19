#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:11:49 2017

@author: robertmarsland
"""

import numpy as np
import models

def MixPairs(CommunityInstance,R0_mix = None):
    n_demes = CommunityInstance.A
    
    #Prepare initial conditions:
    N0_mix = np.zeros((CommunityInstance.S,n_demes**2))
    N0_mix[:,:n_demes] = CommunityInstance.N
    if R0_mix == None:
        R0_mix = np.ones((CommunityInstance.M,n_demes**2))
        
    #Make mixing matrix
    f_mix = np.zeros((n_demes**2,n_demes**2))
    f1 = np.zeros((n_demes**2,n_demes))
    f2 = np.zeros((n_demes**2,n_demes))
    m1 = np.eye(n_demes)
    for k in range(n_demes):
        m2 = np.zeros((n_demes,n_demes))
        m2[:,k] = 1
        f1[k*n_demes:n_demes+k*n_demes,:] = m1
        f2[k*n_demes:n_demes+k*n_demes,:] = m2
        f_mix[k*n_demes:n_demes+k*n_demes,:n_demes] = m1 + m2
        
    #Compute initial community compositions and sum    
    N_1 = np.dot(CommunityInstance.N,f1.T)
    N_2 = np.dot(CommunityInstance.N,f2.T)
    N_sum = 0.5*(N_1+N_2)
        
    #Initialize new community and apply mixing
    Batch_mix = CommunityInstance.copy()
    Batch_mix.Reset([N0_mix,R0_mix])
    Batch_mix.Dilute(f_mix)
    
    return Batch_mix, N_1, N_2, N_sum

def BinaryRandomMatrix(a,b,p):
    r = np.random.rand(a,b)
    m = np.zeros((a,b))
    m[r<p] = 1.0
    
    return m

Stot = 1000
Sbar = 100
M = 10
n_demes = 10

def UniformRandomCRM(cmin = 0.7, consume_frac = 0.7, Stot = 1000, Sbar = 100, 
                     M = 10, n_demes = 10):
    off = 1./((1./cmin)-1)
    c = (np.random.rand(Stot,M)+off)/(1+off)*BinaryRandomMatrix(Stot,M,consume_frac)
    m = np.ones(Stot)*0.01
    N0 = BinaryRandomMatrix(Stot,n_demes,Sbar*1./Stot)*1./Sbar
    R0 = np.ones((M,n_demes))
    
    init_state = [N0,R0]
    dynamics = [models.dNdt_CRM,models.dRdt_CRM]
    params = [c,m]
    
    
    return init_state, dynamics, params

def SimpleDilution(CommunityInstance, f0 = 1e-3):
    n_demes = CommunityInstance.A
    f = f0 * np.eye(n_demes)
    return f