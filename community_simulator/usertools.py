#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:11:49 2017

@author: robertmarsland
"""

import numpy as np
import models

def CosDist(df1,df2):
    return (df1*df2).sum()/np.sqrt((df1*df1).sum()*(df2*df2).sum())

def MixPairs(CommunityInstance1, CommunityInstance2, R0_mix = 'Com1'):
    assert np.all(CommunityInstance1.N.index == CommunityInstance2.N.index), "Communities must have the same species names."
    assert np.all(CommunityInstance1.R.index == CommunityInstance2.R.index), "Communities must have the same resource names."
    
    n_demes1 = CommunityInstance1.A
    n_demes2 = CommunityInstance2.A
    
    #Prepare initial conditions:
    N0_mix = np.zeros((CommunityInstance1.S,n_demes1*n_demes2))
    N0_mix[:,:n_demes1] = CommunityInstance1.N
    N0_mix[:,n_demes1:n_demes1+n_demes2] = CommunityInstance2.N
    if type(R0_mix) == str:
        if R0_mix == 'Com1':
            R0vec = CommunityInstance1.R0.iloc[:,0].values[:,np.newaxis]
            R0_mix = np.dot(R0vec,np.ones((1,n_demes1*n_demes2)))
        elif R0_mix == 'Com2':
            R0vec = CommunityInstance2.R0.iloc[:,0].values[:,np.newaxis]
            R0_mix = np.dot(R0vec,np.ones((1,n_demes1*n_demes2)))
    else:
        assert np.shape(R0_mix) == (CommunityInstance1.M,n_demes1*n_demes2), "Valid R0_mix values are 'Com1', 'Com2', or a resource matrix of dimension M x (n_demes1*n_demes2)."
        
    #Make mixing matrix
    f_mix = np.zeros((n_demes1*n_demes2,n_demes1*n_demes2))
    f1 = np.zeros((n_demes1*n_demes2,n_demes1))
    f2 = np.zeros((n_demes1*n_demes2,n_demes2))
    m1 = np.eye(n_demes1)
    for k in range(n_demes2):
        m2 = np.zeros((n_demes1,n_demes2))
        m2[:,k] = 1
        f1[k*n_demes1:n_demes1+k*n_demes1,:] = m1
        f2[k*n_demes1:n_demes1+k*n_demes1,:] = m2
        f_mix[k*n_demes1:n_demes1+k*n_demes1,:n_demes1] = 0.5*m1
        f_mix[k*n_demes1:n_demes1+k*n_demes1,n_demes1:n_demes1+n_demes2] = 0.5*m2
        
    #Compute initial community compositions and sum    
    N_1 = np.dot(CommunityInstance1.N,f1.T)
    N_2 = np.dot(CommunityInstance2.N,f2.T)
    N_sum = 0.5*(N_1+N_2)
        
    #Initialize new community and apply mixing
    Batch_mix = CommunityInstance1.copy()
    Batch_mix.Reset([N0_mix,R0_mix])
    Batch_mix.Dilute(f_mix,include_resource=False)
    
    return Batch_mix, N_1, N_2, N_sum

def BinaryRandomMatrix(a,b,p):
    r = np.random.rand(a,b)
    m = np.zeros((a,b))
    m[r<p] = 1.0
    
    return m

def UniformRandomCRM(cmin = 0.7, consume_frac = 0.7, Stot = 1000, Sbar = 100, 
                     M = 10, n_demes = 10, main_resource_ind = 0,
                     trace_resource_abund = 1.0e-3):
    off = 1./((1./cmin)-1)
    c = (np.random.rand(Stot,M)+off)/(1+off)*BinaryRandomMatrix(Stot,M,consume_frac)
    m = np.ones(Stot)*0.01
    N0 = BinaryRandomMatrix(Stot,n_demes,Sbar*1./Stot)*1./Sbar
    R0 = np.ones((M,n_demes))*trace_resource_abund
    R0[main_resource_ind] = 1
    
    init_state = [N0,R0]
    dynamics = [models.dNdt_CRM,models.dRdt_CRM]
    params = [c,m]
    
    
    return init_state, dynamics, params

def SimpleDilution(CommunityInstance, f0 = 1e-3):
    n_demes = CommunityInstance.A
    f = f0 * np.eye(n_demes)
    return f