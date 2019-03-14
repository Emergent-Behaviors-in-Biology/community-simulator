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
mp_default = {'sampling':'Binary', #{'Gaussian','Binary','Gamma'} specifies choice of sampling algorithm
          'SA': 60*np.ones(3), #Number of species in each family
          'MA': 30*np.ones(3), #Number of resources of each type
          'Sgen': 30, #Number of generalist species
          'muc': 10, #Mean sum of consumption rates in Gaussian model
          'sigc': 3, #Standard deviation of sum of consumption rates in Gaussian model
          'q': 0.75, #Preference strength (0 for generalist and 1 for specialist)
          'c0':0.01, #Background consumption rate in binary model
          'c1':1., #Specific consumption rate in binary model
          'fs':0.3, #Fraction of secretion flux with same resource type
          'fw':0.6, #Fraction of secretion flux to 'waste' resource
          'D_diversity':0.2, #Variability in secretion fluxes among resources (must be less than 1)
          'n_wells':10, #Number of independent wells
          'S':100, #Number of species per well
          'food':0, #index of food source
          'R0_food':1000, #unperturbed fixed point for supplied food
          'regulation':'independent', #metabolic regulation (see dRdt)
          'response':'type I', #functional response (see dRdt)
          'replenishment':'off' #resource replenishment (see dRdt)
         }

def MakeInitialState(metaparams):
    """
    Construct stochastically colonized initial state, at unperturbed resource fixed point.
    
    metaparams = dictionary of metaparameters
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

    M = int(np.sum(metaparams['MA']))
    S_tot = int(np.sum(metaparams['SA']))+metaparams['Sgen']
    R0 = np.zeros((M,metaparams['n_wells']))
    N0 = np.zeros((S_tot,metaparams['n_wells']))
    
    if not isinstance(metaparams['food'],int):
        assert len(metaparams['food']) == metaparams['n_wells'], 'Length of food vector must equal n_wells.'
        food_list = metaparams['food']
    else:
        food_list = np.ones(metaparams['n_wells'],dtype=int)*metaparams['food']

    if not isinstance(metaparams['R0_food'],int):
        assert len(metaparams['R0_food']) == metaparams['n_wells'], 'Length of food vector must equal n_wells.'
        R0_food_list = metaparams['R0_food']
    else:
        R0_food_list = np.ones(metaparams['n_wells'],dtype=int)*metaparams['R0_food']

    for k in range(metaparams['n_wells']):
        N0[np.random.choice(S_tot,size=metaparams['S'],replace=False),k]=1.
        R0[food_list[k],k] = R0_food_list[k]
    
    return N0, R0

def MakeMatrices(metaparams):
    """
    Construct consumer matrix and metabolic matrix.
    
    metaparams = dictionary of metaparameters
        'sampling' = {'Gaussian','Binary','Gamma'} specifies choice of sampling algorithm
        'SA' = number of species in each family
        'MA' = number of resources of each type
        'Sgen' = number of generalist species
        'muc' = mean sum of consumption rates
        'sigc' = standard deviation for Gaussian sampling of consumer matrix
        'q' = family preference strength (from 0 to 1)
        'c0' = background consumption rate for Binary sampling
        'c1' = specific consumption rate for Binary sampling
        'fs' = fraction of secretion flux into same resource type
        'fw' = fraction of secretion flux into waste resource type
        'D_diversity' = variability in secretion fluxes among resources (from 0 to 1)
        'wate_type' = index of resource type to designate as "waste"
    
    Returns:
    c = consumer matrix
    D = metabolic matrix
    """
    #PREPARE VARIABLES
    #Force number of species to be an array:
    if isinstance(metaparams['MA'],numbers.Number):
        metaparams['MA'] = [metaparams['MA']]
    if isinstance(metaparams['SA'],numbers.Number):
        metaparams['SA'] = [metaparams['SA']]
    #Force numbers of species to be integers:
    metaparams['MA'] = np.asarray(metaparams['MA'],dtype=int)
    metaparams['SA'] = np.asarray(metaparams['SA'],dtype=int)
    metaparams['Sgen'] = int(metaparams['Sgen'])
    #Default waste type is last type in list:
    if 'waste_type' not in metaparams.keys():
        metaparams['waste_type']=len(metaparams['MA'])-1

    #Extract total numbers of resources, consumers, resource types, and consumer families:
    M = np.sum(metaparams['MA'])
    T = len(metaparams['MA'])
    S = np.sum(metaparams['SA'])+metaparams['Sgen']
    F = len(metaparams['SA'])
    M_waste = metaparams['MA'][metaparams['waste_type']]
    #Construct lists of names of resources, consumers, resource types, and consumer families:
    resource_names = ['R'+str(k) for k in range(M)]
    type_names = ['T'+str(k) for k in range(T)]
    family_names = ['F'+str(k) for k in range(F)]
    consumer_names = ['S'+str(k) for k in range(S)]
    waste_name = type_names[metaparams['waste_type']]
    resource_index = [[type_names[m] for m in range(T) for k in range(metaparams['MA'][m])],
                      resource_names]
    consumer_index = [[family_names[m] for m in range(F) for k in range(metaparams['SA'][m])]
                      +['GEN' for k in range(metaparams['Sgen'])],consumer_names]
    
    
    #PERFORM GAUSSIAN SAMPLING
    if metaparams['sampling'] == 'Gaussian':
        #Sample Gaussian random numbers with standard deviation sigc over sqrt M:
        c = pd.DataFrame(np.random.randn(S,M)*metaparams['sigc']/np.sqrt(M),
                     columns=resource_index,index=consumer_index)
        #Add mean values, biasing consumption of each family towards its preferred resource:
        for k in range(F):
            for j in range(T):
                if k==j:
                    c.loc['F'+str(k)]['T'+str(j)] = c.loc['F'+str(k)]['T'+str(j)].values + (metaparams['muc']/M)*(1+metaparams['q']*(M-metaparams['MA'][j])/metaparams['MA'][j])
                else:
                    c.loc['F'+str(k)]['T'+str(j)] = c.loc['F'+str(k)]['T'+str(j)].values + (metaparams['muc']/M)*(1-metaparams['q'])
        if 'GEN' in c.index:
            c.loc['GEN'] = c.loc['GEN'].values + (metaparams['muc']/M)
                    
    #PERFORM BINARY SAMPLING
    elif metaparams['sampling'] == 'Binary':
        assert metaparams['muc'] < M*metaparams['c1'], 'muc not attainable with given M and c1.'
        #Construct uniform matrix at background consumption rate c0:
        c = pd.DataFrame(np.ones((S,M))*metaparams['c0'],columns=resource_index,index=consumer_index)
        #Sample binary random matrix blocks for each pair of family/resource type:
        for k in range(F):
            for j in range(T):
                if k==j:
                    p = (metaparams['muc']/(M*metaparams['c1']))*(1+metaparams['q']*(M-metaparams['MA'][j])/metaparams['MA'][j])
                else:
                    p = (metaparams['muc']/(M*metaparams['c1']))*(1-metaparams['q'])
                    
                c.loc['F'+str(k)]['T'+str(j)] = (c.loc['F'+str(k)]['T'+str(j)].values 
                                                + metaparams['c1']*BinaryRandomMatrix(metaparams['SA'][k],metaparams['MA'][j],p))
        #Sample uniform binary random matrix for generalists:
        if 'GEN' in c.index:
            p = metaparams['muc']/(M*metaparams['c1'])
            c.loc['GEN'] = c.loc['GEN'].values + metaparams['c1']*BinaryRandomMatrix(metaparams['Sgen'],M,p)
    elif metaparams['sampling'] == 'Gamma':
        #Initialize empty dataframe
        c = pd.DataFrame(np.zeros((S,M)),columns=resource_index,index=consumer_index)

        #Add Gamma-sampled values, biasing consumption of each family towards its preferred resource
        for k in range(F):
            for j in range(T):
                if k==j:
                    c_mean = (metaparams['muc']/M)*(1+metaparams['q']*(M-metaparams['MA'][j])/metaparams['MA'][j])
                    thetac = metaparams['sigc']**2*1./(M*c_mean)
                    kc = M*c_mean**2*1./(metaparams['sigc']**2)
                    c.loc['F'+str(k)]['T'+str(j)] = np.random.gamma(kc,scale=thetac,size=(metaparams['SA'][k],metaparams['MA'][j]))
                else:
                    c_mean = (metaparams['muc']/M)*(1-metaparams['q'])
                    thetac = metaparams['sigc']**2*1./(M*c_mean)
                    kc = M*c_mean**2*1./(metaparams['sigc']**2)
                    c.loc['F'+str(k)]['T'+str(j)] = np.random.gamma(kc,scale=thetac,size=(metaparams['SA'][k],metaparams['MA'][j]))
        if 'GEN' in c.index:
            c_mean = metaparams['muc']/M
            thetac = metaparams['sigc']**2*1./(M*c_mean)
            kc = M*c_mean**2*1./(metaparams['sigc']**2)
            c.loc['GEN'] = np.random.gamma(kc,scale=thetac,size=(metaparams['Sgen'],M))
    
    else:
        print('Invalid distribution choice. Valid choices are kind=Gaussian and kind=Binary.')
        return 'Error'

    #SAMPLE METABOLIC MATRIX FROM DIRICHLET DISTRIBUTION
    DT = pd.DataFrame(np.zeros((M,M)),index=c.keys(),columns=c.keys())
    for type_name in type_names:
        MA = len(DT.loc[type_name])
        if type_name is not waste_name:
            #Set background secretion levels
            p = pd.Series(np.ones(M)*(1-metaparams['fs']-metaparams['fw'])/(M-MA-M_waste),index = DT.keys())
            #Set self-secretion level
            p.loc[type_name] = metaparams['fs']/MA
            #Set waste secretion level
            p.loc[waste_name] = metaparams['fw']/M_waste
            #Sample from dirichlet
            DT.loc[type_name] = dirichlet(p/metaparams['D_diversity'],size=MA)
        else:
            if M > MA:
                #Set background secretion levels
                p = pd.Series(np.ones(M)*(1-metaparams['fw']-metaparams['fs'])/(M-MA),index = DT.keys())
                #Set self-secretion level
                p.loc[type_name] = (metaparams['fw']+metaparams['fs'])/MA
            else:
                p = pd.Series(np.ones(M)/M,index = DT.keys())
            #Sample from dirichlet
            DT.loc[type_name] = dirichlet(p/metaparams['D_diversity'],size=MA)
        
    return c, DT.T

def AddLabels(N0_values,R0_values,c):
    """
    Apply labels from consumer matrix to state variables.
    
    c = consumer matrix (as Pandas Data Frame)
    
    N0_values, R0_values = 2D arrays of initial consumer and resource concentrations
    
    Returns:
    N0, R0 = Pandas Data Frames with consumer and resource labels from c.
    """
    
    assert type(c) == pd.DataFrame, 'Consumer matrix must be a Data Frame.'
    
    n_wells = np.shape(N0_values)[1]
    well_names = ['W'+str(k) for k in range(n_wells)]
    N0 = pd.DataFrame(N0_values,columns=well_names,index=c.index)
    R0 = pd.DataFrame(R0_values,columns=well_names,index=c.keys())
    
    return N0, R0

def MakeResourceDynamics(metaparams):
    """
    Construct resource dynamics. 'metaparams' must be a dictionary containing at least
    three entries:
    
    response = {'type I', 'type II', 'type III'} specifies nonlinearity of growth law
    
    regulation = {'independent','energy','mass'} allows microbes to adjust uptake
        rates to favor the most abundant accessible resources (measured either by
        energy or mass)
    
    replenishment = {'off','external','self-renewing'} sets choice of
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
         'self-renewing': lambda R,params: params['r']*R*(params['R0']-R)
    }
    
    J_in = lambda R,params: (u[metaparams['regulation']](params['c']*R,params)
                             *params['w']*sigma[metaparams['response']](R,params))
    J_out = lambda R,params: (params['l']*J_in(R,params)).dot(params['D'].T)
    
    return lambda N,R,params: (h[metaparams['replenishment']](R,params)
                               -(J_in(R,params)/params['w']).T.dot(N)
                               +(J_out(R,params)/params['w']).T.dot(N))

def MakeConsumerDynamics(metaparams):
    """
    Construct resource dynamics. 'metaparams' must be a dictionary containing at least
    three entries:
    
    response = {'type I', 'type II', 'type III'} specifies nonlinearity of growth law
    
    regulation = {'independent','energy','mass'} allows microbes to adjust uptake
        rates to favor the most abundant accessible resources (measured either by
        energy or mass)
    
    replenishment = {'off','external','self-renewing','predator'} sets choice of 
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
    
    J_in = lambda R,params: (u[metaparams['regulation']](params['c']*R,params)
                             *params['w']*sigma[metaparams['response']](R,params))
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

