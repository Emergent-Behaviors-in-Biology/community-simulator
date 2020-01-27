#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:11:49 2017

@author: robertmarsland
"""
from __future__ import division
import numpy as np
import pandas as pd
from numpy.random import dirichlet
import numbers

#DEFAULT PARAMETERS FOR CONSUMER AND METABOLIC MATRICES, AND INITIAL STATE
a_default = {'sampling':'Binary', #{'Gaussian','Binary','Gamma'} specifies choice of sampling algorithm
          'SA': 60*np.ones(3), #Number of species in each specialist family (here, 3 families of 60 species)
          'MA': 30*np.ones(3), #Number of resources in each class 
          'Sgen': 30, #Number of generalist species (unbiased sampling over alll resource classes)
          'muc': 10, #Mean sum of consumption rates (used in all models)
          'sigc': 3, #Standard deviation of sum of consumption rates for Gaussian and Gamma models
          'q': 0.0, #Preference strength of specialist families (0 for generalist and 1 for specialist)
          'c0':0.0, #Sum of background consumption rates in binary model
          'c1':1., #Specific consumption rate in binary model
          'l':0.8, #Leakage fraction
          'fs':0.45, #Fraction of secretion flux with same resource type
          'fw':0.45, #Fraction of secretion flux to 'waste' resource
          'sparsity':0.2, #Effective sparsity of metabolic matrix (between 0 and 1)
          'n_wells':10, #Number of independent wells
          'S':100, #Number of species per well (randomly sampled from the pool of size Stot = sum(SA) + Sgen)
          'food':0, #index of food source (when a single resource is supplied externally)
          'R0_food':1000, #unperturbed fixed point for supplied food
          'regulation':'independent', #metabolic regulation (see dRdt)
          'response':'type I', #functional response (see dRdt)
          'supply':'off' #resource supply (see dRdt)
         }

def MakeInitialState(assumptions):
    """
    Construct stochastically colonized initial state, at unperturbed resource fixed point.
    
    assumptions = dictionary of metaparameters
        'SA' = number of species in each family
        'MA' = number of resources of each type
        'Sgen' = number of generalist species
        'n_wells' = number of independent wells in the experiment
        'S' = initial number of species per well
        'food' = index of supplied "food" resource
        'R0_food' = unperturbed fixed point for supplied food resource
    
    Returns:
    N0 = initial consumer populations
    R0 = initial resource concentrations
    """

    #PREPARE VARIABLES
    #Force number of species to be an array:
    if isinstance(assumptions['MA'],numbers.Number):
        assumptions['MA'] = [assumptions['MA']]
    if isinstance(assumptions['SA'],numbers.Number):
        assumptions['SA'] = [assumptions['SA']]
    #Force numbers of species to be integers:
    assumptions['MA'] = np.asarray(assumptions['MA'],dtype=int)
    assumptions['SA'] = np.asarray(assumptions['SA'],dtype=int)
    assumptions['Sgen'] = int(assumptions['Sgen'])

    #Extract total numbers of resources, consumers, resource types, and consumer families:
    M = int(np.sum(assumptions['MA']))
    T = len(assumptions['MA'])
    S_tot = int(np.sum(assumptions['SA'])+assumptions['Sgen'])
    F = len(assumptions['SA'])
    #Construct lists of names of resources, consumers, resource types, consumer families and wells:
    resource_names = ['R'+str(k) for k in range(M)]
    type_names = ['T'+str(k) for k in range(T)]
    family_names = ['F'+str(k) for k in range(F)]
    consumer_names = ['S'+str(k) for k in range(S_tot)]
    resource_index = [[type_names[m] for m in range(T) for k in range(assumptions['MA'][m])],
                      resource_names]
    consumer_index = [[family_names[m] for m in range(F) for k in range(assumptions['SA'][m])]
                      +['GEN' for k in range(assumptions['Sgen'])],consumer_names]
    well_names = ['W'+str(k) for k in range(assumptions['n_wells'])]

    R0 = np.zeros((M,assumptions['n_wells']))
    N0 = np.zeros((S_tot,assumptions['n_wells']))
    
    if not isinstance(assumptions['food'],int):
        assert len(assumptions['food']) == assumptions['n_wells'], 'Length of food vector must equal n_wells.'
        food_list = assumptions['food']
    else:
        food_list = np.ones(assumptions['n_wells'],dtype=int)*assumptions['food']

    if not (isinstance(assumptions['R0_food'],int) or isinstance(assumptions['R0_food'],float)):
        assert len(assumptions['R0_food']) == assumptions['n_wells'], 'Length of food vector must equal n_wells.'
        R0_food_list = assumptions['R0_food']
    else:
        R0_food_list = np.ones(assumptions['n_wells'],dtype=int)*assumptions['R0_food']

    for k in range(assumptions['n_wells']):
        N0[np.random.choice(S_tot,size=assumptions['S'],replace=False),k]=1.
        R0[food_list[k],k] = R0_food_list[k]

    N0 = pd.DataFrame(N0,index=consumer_index,columns=well_names)
    R0 = pd.DataFrame(R0,index=resource_index,columns=well_names)

    return N0, R0

def MakeMatrices(assumptions):
    """
    Construct consumer matrix and metabolic matrix.
    
    assumptions = dictionary of metaparameters
        'sampling' = {'Gaussian','Binary','Gamma'} specifies choice of sampling algorithm
        'SA' = number of species in each family
        'MA' = number of resources of each type
        'Sgen' = number of generalist species
        'muc' = mean sum of consumption rates
        'sigc' = standard deviation for Gaussian sampling of consumer matrix
        'q' = family preference strength (from 0 to 1)
        'c0' = row sum of background consumption rates for Binary sampling
        'c1' = specific consumption rate for Binary sampling
        'fs' = fraction of secretion flux into same resource type
        'fw' = fraction of secretion flux into waste resource type
        'sparsity' = effective sparsity of metabolic matrix (from 0 to 1)
        'wate_type' = index of resource type to designate as "waste"
    
    Returns:
    c = consumer matrix
    D = metabolic matrix
    """
    #PREPARE VARIABLES
    #Force number of species to be an array:
    if isinstance(assumptions['MA'],numbers.Number):
        assumptions['MA'] = [assumptions['MA']]
    if isinstance(assumptions['SA'],numbers.Number):
        assumptions['SA'] = [assumptions['SA']]
    #Force numbers of species to be integers:
    assumptions['MA'] = np.asarray(assumptions['MA'],dtype=int)
    assumptions['SA'] = np.asarray(assumptions['SA'],dtype=int)
    assumptions['Sgen'] = int(assumptions['Sgen'])
    #Default waste type is last type in list:
    if 'waste_type' not in assumptions.keys():
        assumptions['waste_type']=len(assumptions['MA'])-1

    #Extract total numbers of resources, consumers, resource types, and consumer families:
    M = np.sum(assumptions['MA'])
    T = len(assumptions['MA'])
    S = np.sum(assumptions['SA'])+assumptions['Sgen']
    F = len(assumptions['SA'])
    M_waste = assumptions['MA'][assumptions['waste_type']]
    #Construct lists of names of resources, consumers, resource types, and consumer families:
    resource_names = ['R'+str(k) for k in range(M)]
    type_names = ['T'+str(k) for k in range(T)]
    family_names = ['F'+str(k) for k in range(F)]
    consumer_names = ['S'+str(k) for k in range(S)]
    waste_name = type_names[assumptions['waste_type']]
    resource_index = [[type_names[m] for m in range(T) for k in range(assumptions['MA'][m])],
                      resource_names]
    consumer_index = [[family_names[m] for m in range(F) for k in range(assumptions['SA'][m])]
                      +['GEN' for k in range(assumptions['Sgen'])],consumer_names]
    
    #PERFORM GAUSSIAN SAMPLING
    if assumptions['sampling'] == 'Gaussian':
        #Initialize dataframe:
        c = pd.DataFrame(np.zeros((S,M)),columns=resource_index,index=consumer_index)
        #Add Gaussian-sampled values, biasing consumption of each family towards its preferred resource:
        for k in range(F):
            for j in range(T):
                if k==j:
                    c_mean = (assumptions['muc']/M)*(1+assumptions['q']*(M-assumptions['MA'][j])/assumptions['MA'][j])
                    c_var = (assumptions['sigc']**2/M)*(1+assumptions['q']*(M-assumptions['MA'][j])/assumptions['MA'][j])
                else:
                    c_mean = (assumptions['muc']/M)*(1-assumptions['q'])
                    c_var = (assumptions['sigc']**2/M)*(1-assumptions['q'])
                c.loc['F'+str(k)]['T'+str(j)] = c_mean + np.random.randn(assumptions['SA'][k],assumptions['MA'][j])*np.sqrt(c_var)
        if 'GEN' in c.index:
            c_mean = assumptions['muc']/M
            c_var = assumptions['sigc']**2/M
            c.loc['GEN'] = c_mean + np.random.randn(assumptions['Sgen'],M)*np.sqrt(c_var)
                    
    #PERFORM BINARY SAMPLING
    elif assumptions['sampling'] == 'Binary':
        assert assumptions['muc'] < M*assumptions['c1'], 'muc not attainable with given M and c1.'
        #Construct uniform matrix at total background consumption rate c0:
        c = pd.DataFrame(np.ones((S,M))*assumptions['c0']/M,columns=resource_index,index=consumer_index)
        #Sample binary random matrix blocks for each pair of family/resource type:
        for k in range(F):
            for j in range(T):
                if k==j:
                    p = (assumptions['muc']/(M*assumptions['c1']))*(1+assumptions['q']*(M-assumptions['MA'][j])/assumptions['MA'][j])
                else:
                    p = (assumptions['muc']/(M*assumptions['c1']))*(1-assumptions['q'])
                    
                c.loc['F'+str(k)]['T'+str(j)] = (c.loc['F'+str(k)]['T'+str(j)].values 
                                                + assumptions['c1']*BinaryRandomMatrix(assumptions['SA'][k],assumptions['MA'][j],p))
        #Sample uniform binary random matrix for generalists:
        if 'GEN' in c.index:
            p = assumptions['muc']/(M*assumptions['c1'])
            c.loc['GEN'] = c.loc['GEN'].values + assumptions['c1']*BinaryRandomMatrix(assumptions['Sgen'],M,p)

    elif assumptions['sampling'] == 'Gamma':
        #Initialize dataframe
        c = pd.DataFrame(np.zeros((S,M)),columns=resource_index,index=consumer_index)
        #Add Gamma-sampled values, biasing consumption of each family towards its preferred resource
        for k in range(F):
            for j in range(T):
                if k==j:
                    c_mean = (assumptions['muc']/M)*(1+assumptions['q']*(M-assumptions['MA'][j])/assumptions['MA'][j])
                    c_var = (assumptions['sigc']**2/M)*(1+assumptions['q']*(M-assumptions['MA'][j])/assumptions['MA'][j])
                    thetac = c_var/c_mean
                    kc = c_mean**2/c_var
                    c.loc['F'+str(k)]['T'+str(j)] = np.random.gamma(kc,scale=thetac,size=(assumptions['SA'][k],assumptions['MA'][j]))
                else:
                    c_mean = (assumptions['muc']/M)*(1-assumptions['q'])
                    c_var = (assumptions['sigc']**2/M)*(1-assumptions['q'])
                    thetac = c_var/c_mean
                    kc = c_mean**2/c_var
                    c.loc['F'+str(k)]['T'+str(j)] = np.random.gamma(kc,scale=thetac,size=(assumptions['SA'][k],assumptions['MA'][j]))
        if 'GEN' in c.index:
            c_mean = assumptions['muc']/M
            c_var = assumptions['sigc']**2/M
            thetac = c_var/c_mean
            kc = c_mean**2/c_var
            c.loc['GEN'] = np.random.gamma(kc,scale=thetac,size=(assumptions['Sgen'],M))
        #PERFORM GAUSSIAN SAMPLING
    elif assumptions['sampling'] == 'Uniform':
        #Initialize dataframe:
        c = pd.DataFrame(np.zeros((S,M)),columns=resource_index,index=consumer_index)
        #Add uniformly sampled values, biasing consumption of each family towards its preferred resource:
        for k in range(F):
            for j in range(T):
                if k==j:
                    c_mean = (assumptions['muc']/M)*(1+assumptions['q']*(M-assumptions['MA'][j])/assumptions['MA'][j])
                else:
                    c_mean = (assumptions['muc']/M)*(1-assumptions['q'])
                c.loc['F'+str(k)]['T'+str(j)] = c_mean + (np.random.rand(assumptions['SA'][k],assumptions['MA'][j])-0.5)*assumptions['b']
        if 'GEN' in c.index:
            c_mean = assumptions['muc']/M
            c.loc['GEN'] = c_mean + (np.random.rand(assumptions['Sgen'],M)-0.5)*assumptions['b']
    
    else:
        print('Invalid distribution choice. Valid choices are kind=Gaussian, kind=Binary, kind=Gamma, kind=Uniform.')
        return 'Error'

    #SAMPLE METABOLIC MATRIX FROM DIRICHLET DISTRIBUTION
    DT = pd.DataFrame(np.zeros((M,M)),index=c.keys(),columns=c.keys())
    for type_name in type_names:
        MA = len(DT.loc[type_name])
        if type_name is not waste_name:
            #Set background secretion levels
            p = pd.Series(np.ones(M)*(1-assumptions['fs']-assumptions['fw'])/(M-MA-M_waste),index = DT.keys())
            #Set self-secretion level
            p.loc[type_name] = assumptions['fs']/MA
            #Set waste secretion level
            p.loc[waste_name] = assumptions['fw']/M_waste
            #Sample from dirichlet
            DT.loc[type_name] = dirichlet(p/assumptions['sparsity'],size=MA)
        else:
            if M > MA:
                #Set background secretion levels
                p = pd.Series(np.ones(M)*(1-assumptions['fw']-assumptions['fs'])/(M-MA),index = DT.keys())
                #Set self-secretion level
                p.loc[type_name] = (assumptions['fw']+assumptions['fs'])/MA
            else:
                p = pd.Series(np.ones(M)/M,index = DT.keys())
            #Sample from dirichlet
            DT.loc[type_name] = dirichlet(p/assumptions['sparsity'],size=MA)
        
    return c, DT.T

def MakeParams(assumptions):
    """
    Makes a dictionary of parameters, using MakeMatrices for the matrices, MakeInitialState
    for the resource supply point, and setting everything else to 1, except l which is zero.
    
    Parameter values can be modified from 1 (or zero for l) by adding their name-value pairs
    to the assumptions dictionary.
    """

    c, D = MakeMatrices(assumptions)
    N0,R0 = MakeInitialState(assumptions)
    
    if not isinstance(assumptions['food'],int) or not isinstance(assumptions['R0_food'],int):
        params=[{'c':c,
                'm':1,
                'w':1,
                'D':D,
                'g':1,
                'l':0,
                'R0':R0.values[:,k],
                'tau':1,
                'r':1,
                'sigma_max':1,
                'nreg':10,
                'n':2
                } for k in range(assumptions['n_wells'])]
        for item in ['m','w','g','l','tau','r','sigma_max','n','nreg']:
            if item in assumptions.keys():
                for k in range(assumptions['n_wells']):
                    params[k][item] = assumptions[item]

    else:
        params={'c':c,
                'm':1,
                'w':1,
                'D':D,
                'g':1,
                'l':0,
                'R0':R0.values[:,0],
                'tau':1,
                'r':1,
                'sigma_max':1,
                'nreg':10,
                'n':2
                }
            
        for item in ['m','w','g','l','tau','r','sigma_max','n','nreg']:
            if item in assumptions.keys():
                params[item] = assumptions[item]

    return params

def MakeResourceDynamics(assumptions):
    """
    Construct resource dynamics. 'assumptions' must be a dictionary containing at least
    three entries:
    
    response = {'type I', 'type II', 'type III'} specifies nonlinearity of growth law
    
    regulation = {'independent','energy','mass'} allows microbes to adjust uptake
        rates to favor the most abundant accessible resources (measured either by
        energy or mass)
    
    supply = {'off','external','self-renewing'} sets choice of
        intrinsic resource dynamics
        
    Returns a function of N, R, and the model parameters, which itself returns the
        vector of resource rates of change dR/dt
    """
    sigma = {'type I': lambda R,params: params['c']*R,
             'type II': lambda R,params: params['c']*R/(1+params['c']*R/params['sigma_max']),
             'type III': lambda R,params: (params['c']*R)**params['n']/(1+(params['c']*R)**params['n']/params['sigma_max'])
        }
    
    u = {'independent': lambda x,params: 1.,
         'energy': lambda x,params: (((params['w']*x)**params['nreg']).T
                                      /np.sum((params['w']*x)**params['nreg'],axis=1)).T,
         'mass': lambda x,params: ((x**params['nreg']).T/np.sum(x**params['nreg'],axis=1)).T
        }
    
    h = {'off': lambda R,params: 0.,
         'external': lambda R,params: (params['R0']-R)/params['tau'],
         'self-renewing': lambda R,params: params['r']*R*(params['R0']-R),
         'predator': lambda R,params: params['r']*R*(params['R0']-R)-params['u']*R
    }
    
    J_in = lambda R,params: (u[assumptions['regulation']](params['c']*R,params)
                             *params['w']*sigma[assumptions['response']](R,params))
    J_out = lambda R,params: (params['l']*J_in(R,params)).dot(params['D'].T)
    
    return lambda N,R,params: (h[assumptions['supply']](R,params)
                               -(J_in(R,params)/params['w']).T.dot(N)
                               +(J_out(R,params)/params['w']).T.dot(N))

def MakeConsumerDynamics(assumptions):
    """
    Construct resource dynamics. 'assumptions' must be a dictionary containing at least
    three entries:
    
    response = {'type I', 'type II', 'type III'} specifies nonlinearity of growth law
    
    regulation = {'independent','energy','mass'} allows microbes to adjust uptake
        rates to favor the most abundant accessible resources (measured either by
        energy or mass)
    
    supply = {'off','external','self-renewing','predator'} sets choice of
        intrinsic resource dynamics
        
    Returns a function of N, R, and the model parameters, which itself returns the
        vector of consumer rates of change dN/dt
    """
    sigma = {'type I': lambda R,params: params['c']*R,
             'type II': lambda R,params: params['c']*R/(1+params['c']*R/params['sigma_max']),
             'type III': lambda R,params: (params['c']*R)**params['n']/(1+(params['c']*R)**params['n']/params['sigma_max'])
            }
    
    u = {'independent': lambda x,params: 1.,
         'energy': lambda x,params: (((params['w']*x)**params['nreg']).T
                                      /np.sum((params['w']*x)**params['nreg'],axis=1)).T,
         'mass': lambda x,params: ((x**params['nreg']).T/np.sum(x**params['nreg'],axis=1)).T
        }
    
    J_in = lambda R,params: (u[assumptions['regulation']](params['c']*R,params)
                             *params['w']*sigma[assumptions['response']](R,params))
    J_growth = lambda R,params: (1-params['l'])*J_in(R,params)
    
    return lambda N,R,params: params['g']*N*(np.sum(J_growth(R,params),axis=1)-params['m'])

def MixPairs(plate1, plate2, R0_mix = 'Com1'):
    """
    Perform "community coalescence" by mixing pairs of communities.
    
    plate1, plate2 = plates containing communities to be mixed
    
    R0_mix = {'Com1', 'Com2', matrix of dimension Mxn_wells1xn_wells2} specifies
        the resource profile to be supplied to the mixture
    
    Returns:
    plate_mixed = plate containing 50/50 mixtures of all pairs of communities 
        from plate1 and plate2.
    N_1, N_2 = compositions of original communities
    N_sum = initial compositions of mixed communities
    """
    assert np.all(plate1.N.index == plate2.N.index), "Communities must have the same species names."
    assert np.all(plate1.R.index == plate2.R.index), "Communities must have the same resource names."
    
    n_wells1 = plate1.n_wells
    n_wells2 = plate2.n_wells
    
    #Prepare initial conditions:
    N0_mix = np.zeros((plate1.S,n_wells1*n_wells2))
    N0_mix[:,:n_wells1] = plate1.N
    N0_mix[:,n_wells1:n_wells1+n_wells2] = plate2.N
    if type(R0_mix) == str:
        if R0_mix == 'Com1':
            R0vec = plate1.R0.iloc[:,0].values[:,np.newaxis]
            R0_mix = np.dot(R0vec,np.ones((1,n_wells1*n_wells2)))
        elif R0_mix == 'Com2':
            R0vec = plate2.R0.iloc[:,0].values[:,np.newaxis]
            R0_mix = np.dot(R0vec,np.ones((1,n_wells1*n_wells2)))
    else:
        assert np.shape(R0_mix) == (plate1.M,n_wells1*n_wells2), "Valid R0_mix values are 'Com1', 'Com2', or a resource matrix of dimension M x (n_wells1*n_wells2)."
        
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
    N_1 = np.dot(plate1.N,f1.T)
    N_2 = np.dot(plate2.N,f2.T)
    N_sum = 0.5*(N_1+N_2)
        
    #Initialize new community and apply mixing
    plate_mixed = plate1.copy()
    plate_mixed.Reset([N0_mix,R0_mix])
    plate_mixed.Passage(f_mix,include_resource=False)
    
    return plate_mixed, N_1, N_2, N_sum

def SimpleDilution(plate, f0 = 1e-3):
    """
    Returns identity matrix of size plate.n_wells, scaled by f0
    """
    f = f0 * np.eye(plate.n_wells)
    return f

def BinaryRandomMatrix(a,b,p):
    """
    Construct binary random matrix.
    
    a, b = matrix dimensions
    
    p = probability that element equals 1 (otherwise 0)
    """
    r = np.random.rand(a,b)
    m = np.zeros((a,b))
    m[r<p] = 1.0
    
    return m

