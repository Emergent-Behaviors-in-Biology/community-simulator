#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:11:49 2017

@author: robertmarsland
"""

import numpy as np
import pandas as pd
from numpy.random import dirichlet

#Default parameters for consumer matrix
params_default = {'SA': 3*np.ones(3), #Number of species in each family
          'MA': 3*np.ones(3), #Number of resources of each type
          'Sgen': 3, #Number of generalist species
          'muc': 1, #Mean sum of consumption rates in Gaussian model
          'sigc': .01, #Variance in consumption rate in Gaussian model
          'q': 2./3, #Preference strength 
          'c0':0.0001, #Background consumption rate in binary model
          'c1':1., #Maximum consumption rate in binary model
          'fs':0.29, #Fraction of secretion flux with same resource type
          'fw':0.7, #Fraction of secretion flux to 'waste' resource
          'D_diversity':0.001 #Variability in secretion fluxes among resources (must be less than 1)
         }


def MakeMatrices(params = params_default, kind='Gaussian', waste_ind=0):
    """Construct consumer matrix with family structure specified in parameter dictionary params.
    Choose one of two kinds of sampling: Gaussian or Binary.
    waste_ind specifies the index of the resource type to be designated 'waste.'"""
    
    #Force numbers of species to be integers
    params['MA'] = np.asarray(params['MA'],dtype=int)
    params['SA'] = np.asarray(params['SA'],dtype=int)
    params['Sgen'] = int(params['Sgen'])
    
    #Extract total numbers of resources, consumers, resource types, and consumer families
    M = np.sum(params['MA'])
    T = len(params['MA'])
    S = np.sum(params['SA'])+params['Sgen']
    F = len(params['SA'])
    M_waste = params['MA'][waste_ind]
    
    #Construct lists of names of resources, consumers, resource types, and consumer families
    resource_names = ['R'+str(k) for k in range(M)]
    type_names = ['T'+str(k) for k in range(T)]
    family_names = ['F'+str(k) for k in range(F)]
    consumer_names = ['S'+str(k) for k in range(S)]
    waste_name = type_names[waste_ind]
    resource_index = [[type_names[m] for m in range(T) for k in range(params['MA'][m])],
                      resource_names]
    consumer_index = [[family_names[m] for m in range(F) for k in range(params['SA'][m])]
                      +['GEN' for k in range(params['Sgen'])],consumer_names]
    
    
    #Perform Gaussian sampling
    if kind == 'Gaussian':
        #Sample Gaussian random numbers with variance sigc and mean muc/M
        c = pd.DataFrame(np.random.randn(S,M)*params['sigc']+np.ones((S,M))*params['muc']/M,
                     columns=resource_index,index=consumer_index)
    
        #Bias consumption of each family towards its preferred resource
        for k in range(F):
            for j in range(T):
                if k==j:
                    c.loc['F'+str(k),'T'+str(j)] = c.loc['F'+str(k),'T'+str(j)].values + params['q']/params['MA'][k]
                else:
                    c.loc['F'+str(k),'T'+str(j)] = c.loc['F'+str(k),'T'+str(j)].values - params['q']/(M-params['MA'][k])
                    
    #Perform binary sampling
    elif kind == 'Binary':
        #Construct uniform matrix at background consumption rate c0
        c = pd.DataFrame(np.ones((S,M))*params['c0'],columns=resource_index,index=consumer_index)
    
        #Sample binary random matrix blocks for each pair of family/resource type
        for k in range(F):
            for j in range(T):
                if k==j:
                    p = (1./M) + params['q']/params['MA'][k]
                else:
                    p = (1./M) - params['q']/(M-params['MA'][k])
                    
                c.loc['F'+str(k),'T'+str(j)] = (c.loc['F'+str(k),'T'+str(j)].values 
                                                + BinaryRandomMatrix(params['SA'][k],params['MA'][j],p))
        #Sample uniform binary random matrix for generalists  
        p = 1./M
        c.loc['GEN'] = c.loc['GEN'].values + BinaryRandomMatrix(params['Sgen'],M,p)
    
    else:
        print('Invalid distribution choice. Valid choices are kind=Gaussian and kind=Binary.')
        return 'Error'
        
    #Make crossfeeding matrix
    D = pd.DataFrame(np.zeros((M,M)),index=c.keys(),columns=c.keys())
    for type_name in type_names:
        MA = len(D.loc[type_name])
        #Set background secretion levels
        p = pd.Series(np.ones(M)*(1-params['fs']-params['fw'])/(M-MA-M_waste),index = D.keys())
        #Set self-secretion level
        p.loc[type_name] = params['fs']/MA
        #Set waste secretion level
        p.loc[waste_name] = params['fw']/M_waste
        #Sample from dirichlet
        D.loc[type_name] = dirichlet(p/params['D_diversity'],size=MA)
        
    return c, D

def AddLabels(N0_values,R0_values,c):
    """Apply labels from consumer matrix c to arrays of initial consumer and resource 
    concentrations N0_values and R0_values."""
    
    assert type(c) == pd.DataFrame, 'Consumer matrix must be a Data Frame.'
    
    n_wells = np.shape(N0_values)[1]
    well_names = ['W'+str(k) for k in range(n_wells)]
    N0 = pd.DataFrame(N0_values,columns=well_names,index=c.index)
    R0 = pd.DataFrame(R0_values,columns=well_names,index=c.keys())
    
    return N0, R0

def MakeResourceDynamics(response='type I',regulation='independent',replenishment='off'):
    sigma = {'type I': lambda R,params: params['c']*R,
             'type II': lambda R,params: params['c']*R/(1+params['c']*R/params['K']),
             'type III': lambda R,params: params['c']*(R**params['n'])/(1+params['c']*(R**params['n'])/params['K'])
            }
    
    u = {'independent': lambda x,params: 1.,
         'energy': lambda x,params: (((params['w']*x)**params['nreg']).T
                                      /np.sum((params['w']*x)**params['nreg'],axis=1)).T,
         'mass': lambda x,params: ((x**params['nreg']).T/np.sum(x**params['nreg'],axis=1)).T
        }
    
    h = {'off': lambda R,params: 0.,
         'renew': lambda R,params: (params['R0']-R)/params['tau'],
         'non-renew': lambda R,params: params['r']*R*(params['R0']-R)}
    
    F_in = lambda R,params: (u[regulation](params['c']*R,params)
                             *params['w']*sigma[response](R,params))
    F_out = lambda R,params: ((1-params['e'])*F_in(R,params)).dot(params['D'].T)
    
    return lambda N,R,params: (h[replenishment](R,params)
                               -(F_in(R,params)/params['w']).T.dot(N)
                               +(F_out(R,params)/params['w']).T.dot(N))

def MakeConsumerDynamics(response='type I',regulation='independent',replenishment='off'):
    sigma = {'type I': lambda R,params: params['c']*R,
             'type II': lambda R,params: params['c']*R/(1+params['c']*R/params['K']),
             'type III': lambda R,params: params['c']*(R**params['n'])/(1+params['c']*(R**params['n'])/params['K'])
            }
    
    u = {'independent': lambda x,params: 1.,
         'energy': lambda x,params: (((params['w']*x)**params['nreg']).T
                                      /np.sum((params['w']*x)**params['nreg'],axis=1)).T,
         'mass': lambda x,params: ((x**params['nreg']).T/np.sum(x**params['nreg'],axis=1)).T
        }
    
    F_in = lambda R,params: (u[regulation](params['c']*R,params)
                             *params['w']*sigma[response](R,params))
    F_growth = lambda R,params: params['e']*F_in(R,params)
    
    return lambda N,R,params: params['g']*N*(np.sum(F_growth(R,params),axis=1)-params['m'])

def MixPairs(CommunityInstance1, CommunityInstance2, R0_mix = 'Com1'):
    assert np.all(CommunityInstance1.N.index == CommunityInstance2.N.index), "Communities must have the same species names."
    assert np.all(CommunityInstance1.R.index == CommunityInstance2.R.index), "Communities must have the same resource names."
    
    n_wells1 = CommunityInstance1.A
    n_wells2 = CommunityInstance2.A
    
    #Prepare initial conditions:
    N0_mix = np.zeros((CommunityInstance1.S,n_wells1*n_wells2))
    N0_mix[:,:n_wells1] = CommunityInstance1.N
    N0_mix[:,n_wells1:n_wells1+n_wells2] = CommunityInstance2.N
    if type(R0_mix) == str:
        if R0_mix == 'Com1':
            R0vec = CommunityInstance1.R0.iloc[:,0].values[:,np.newaxis]
            R0_mix = np.dot(R0vec,np.ones((1,n_wells1*n_wells2)))
        elif R0_mix == 'Com2':
            R0vec = CommunityInstance2.R0.iloc[:,0].values[:,np.newaxis]
            R0_mix = np.dot(R0vec,np.ones((1,n_wells1*n_wells2)))
    else:
        assert np.shape(R0_mix) == (CommunityInstance1.M,n_wells1*n_wells2), "Valid R0_mix values are 'Com1', 'Com2', or a resource matrix of dimension M x (n_wells1*n_wells2)."
        
    #Make mixing matrix
    f_mix = np.zeros((n_wells1*n_wells2,n_wells1*n_wells2))
    f1 = np.zeros((n_wells1*n_wells2,n_wells1))
    f2 = np.zeros((n_wells1*n_wells2,n_wells2))
    m1 = np.eye(n_wells1)
    for k in range(n_wells2):
        m2 = np.zeros((n_wells1,n_wells2))
        m2[:,k] = 1
        f1[k*n_wells1:n_wells1+k*n_wells1,:] = m1
        f2[k*n_wells1:n_wells1+k*n_wells1,:] = m2
        f_mix[k*n_wells1:n_wells1+k*n_wells1,:n_wells1] = 0.5*m1
        f_mix[k*n_wells1:n_wells1+k*n_wells1,n_wells1:n_wells1+n_wells2] = 0.5*m2
        
    #Compute initial community compositions and sum    
    N_1 = np.dot(CommunityInstance1.N,f1.T)
    N_2 = np.dot(CommunityInstance2.N,f2.T)
    N_sum = 0.5*(N_1+N_2)
        
    #Initialize new community and apply mixing
    Batch_mix = CommunityInstance1.copy()
    Batch_mix.Reset([N0_mix,R0_mix])
    Batch_mix.Passage(f_mix,include_resource=False)
    
    return Batch_mix, N_1, N_2, N_sum

def SimpleDilution(CommunityInstance, f0 = 1e-3):
    f = f0 * np.eye(CommunityInstance.n_wells)
    return f

def BinaryRandomMatrix(a,b,p):
    r = np.random.rand(a,b)
    m = np.zeros((a,b))
    m[r<p] = 1.0
    
    return m

def CosDist(df1,df2):
    return (df1*df2).sum()/np.sqrt((df1*df1).sum()*(df2*df2).sum())

