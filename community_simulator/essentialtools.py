#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:05:55 2017

@author: robertmarsland
"""

import pandas as pd
import numpy as np
from scipy import integrate

def IntegrateWell(CommunityInstance,params,y0,T=1,ns=2,return_all=False,log_time=False,compress_resources=False):
    #MAKE LOGARITHMIC TIME AXIS FOR LONG SINGLE RUNS
    if log_time:
        t = np.exp(np.linspace(0,np.log(T),ns))
    else:
        t = np.linspace(0,T,ns)
    
    #COMPRESS STATE AND PARAMETERS TO GET RID OF EXTINCT SPECIES
    S = np.shape(params['c'])[0]
    M = len(y0)-S
    not_extinct = y0>0
    S_comp = np.sum(not_extinct[:S]) #record the new point dividing species from resources
    if not compress_resources:  #only compress resources if we're running non-renewable dynamics
        not_extinct[S:] = True
    not_extinct_idx = np.where(not_extinct)[0]
    y0_comp = y0[not_extinct]
    params_comp = params.copy()
    params_comp['c']=params_comp['c'][not_extinct[:S],:]
    params_comp['c']=params_comp['c'][:,not_extinct[S:]]
    params_comp['D']=params_comp['D'][not_extinct[S:],:]
    params_comp['D']=params_comp['D'][:,not_extinct[S:]]
    for name in ['m','g','e']:
        if name in params_comp.keys():
            if type(params_comp[name]) == np.ndarray:
                assert len(params_comp[name])==S, 'Invalid length for ' + name
                params_comp[name]=params_comp[name][not_extinct[:S]]
    for name in ['w','r','tau']:
        if name in params_comp.keys():
            if type(params_comp[name]) == np.ndarray:
                assert len(params_comp[name])==M, 'Invalid length for ' + name
                params_comp[name]=params_comp[name][not_extinct[S:]]
                
    #INTEGRATE AND RESTORE STATE VECTOR TO ORIGINAL SIZE
    if return_all:
        out = integrate.odeint(CommunityInstance.dydt,y0_comp,t,args=(params_comp,S_comp),mxstep=10000,atol=1e-4)
        traj = np.zeros((np.shape(out)[0],S+M))
        traj[:,not_extinct_idx] = out
        return t, traj
    else:    
        out = integrate.odeint(CommunityInstance.dydt,y0_comp,t,args=(params_comp,S_comp),mxstep=10000,atol=1e-4)[-1]
        yf = np.zeros(len(y0))
        yf[not_extinct_idx] = out
        return yf
    
def TimeStamp(data,t,group='Well'):
    if group == 'Well':
        data_time = data.copy().T
        mdx = pd.MultiIndex.from_product([[t],data_time.index],names=['Time','Well'])
    elif group == 'Species':    
        data_time = data.copy()
        mdx = pd.MultiIndex.from_product([[t],data.index],names=['Time','Species'])
    else:
        return 'Invalid group choice'
        
    data_time.index = mdx
    return data_time