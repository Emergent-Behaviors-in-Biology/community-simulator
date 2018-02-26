#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:25:50 2017

@author: robertmarsland
"""

from community_simulator import Community,usertools
import numpy as np
import pandas as pd

#CONSTRUCT DYNAMICS
assumptions = {'regulation':'independent','replenishment':'renew','response':'type I'}
def dNdt(N,R,params):
    return usertools.MakeConsumerDynamics(**assumptions)(N,R,params)
def dRdt(N,R,params):
    return usertools.MakeResourceDynamics(**assumptions)(N,R,params)
dynamics = [dNdt,dRdt]

def RunCommunity(K=500.,q=0.,e=0.2,fs=0.25,fw=0.25,food=0,Ddiv=0.2,n_types=4,c1=1,
                 MA=25,SA=40,Sgen=40,S=100,n_iter=200,T=5,n_wells=27,run_number=0,
                 params=None,N0=None,extra_time=False):
    """
    Generate communities and grow on a single externally supplied nutrient.
    
    K = chemostat set point for external nutrient supply
    
    q = family preference strength
    
    e = efficiency
    
    fs = fraction of secreted flux to same resource type
    
    fw = fraction of secreted flux to waste resource type
    
    food = index of externally supplied resource
    
    Ddiv = diversity parameter for D matrix
    
    n_types = number of food types
    
    c1 = specific consumption rate for binary sampling
    
    MA = number of resource species per type
    
    SA = number of consumer species per family
    
    Sgen = number of generalist species
    
    S = number of consumer species to initialize in each well
    
    n_iter = number of propagate/passage cycles to run
    
    T = amount of time to propagate before zeroing out low concentrations
    
    n_wells = number of wels
    
    run_number = index for keeping track of multiple runs
    
    params = model parameters (will re-sample if None)
    
    N0 = initial consumer concentrations (will re-sample if None)
    
    extra_time = add several propagation cycles of duration 10, and then 1000,
        to make sure slow modes have converged
    """
    
    #PREPARE VARIABLES
    #Make sure MA, SA, Sgen and S are integers:
    MA = int(round(MA))
    SA = int(round(SA))
    Sgen = int(round(Sgen))
    S=int(round(S))
    #Specify metaparameters:
    sample_par = {'SA': SA*np.ones(n_types), #Number of species in each family
          'MA': MA*np.ones(n_types), #Number of resources of each type
          'Sgen': Sgen, #Number of generalist species
          'muc': 10, #Mean sum of consumption rates
          'q': q, #Preference strength 
          'c0':0.01, #Background consumption rate in binary model
          'c1':c1, #Maximum consumption rate in binary model
          'fs':fs, #Fraction of secretion flux with same resource type
          'fw':fw, #Fraction of secretion flux to 'waste' resource
          'D_diversity':0.2, #Variability in secretion fluxes among resources (must be less than 1)
          'waste_type':n_types-1
         }
    
    #DEFINE INITIAL CONDITIONS
    #Create resource vector and set food supply:
    M = int(np.sum(sample_par['MA']))
    R0 = np.zeros((M,n_wells))
    R0[food,:] = K
    #Create initial conditions for consumers (sub-sampling from regional species pool):
    S_tot = int(np.sum(sample_par['SA']))+sample_par['Sgen']
    if N0 is None:
        N0 = np.zeros((S_tot,n_wells))
        for k in range(n_wells):
            N0[np.random.choice(S_tot,size=S,replace=False),k]=1.
            
    #SAMPLE PARAMETERS
    if params is None:
        c, D = usertools.MakeMatrices(metaparams=sample_par, kind='Binary')
        params={'c':c,
                'm':np.ones(S_tot)+0.1*np.random.randn(S_tot),
                'w':np.ones(M),
                'D':D,
                'g':np.ones(S_tot),
                'e':e,
                'r':1.,
                'tau':1
                }
    else:
        params['e'] = e
        
    #INITIALIZE COMMUNITY
    N0,R0 = usertools.AddLabels(N0,R0,params['c'])
    init_state = [N0,R0]
    params['R0']=R0.values[:,0]
    MyPlate = Community(init_state,dynamics,params)
    
    #SIMULATE
    Ntraj,Rtraj = MyPlate.RunExperiment(np.eye(n_wells),T,n_iter,refresh_resource=False,scale=1e6)
    if extra_time:
        Ntraj2,Rtraj2 = MyPlate.RunExperiment(np.eye(n_wells),100,10,refresh_resource=False,scale=1e6)
        Ntraj2,Rtraj2 = MyPlate.RunExperiment(np.eye(n_wells),1000,10,refresh_resource=False,scale=1e6)
    MyPlate.Passage(np.eye(n_wells),refresh_resource=False,scale=1e6)
    
    #EXTRACT RICHNESS AND FINAL STATE
    richness = np.mean((MyPlate.N>0).sum().values)
    final_state = [MyPlate.N.copy(), MyPlate.R.copy()]
    #Stamp final state with run number using Pandas multiindex:
    for j in range(2):
        final_state[j]['Run Number']=run_number
        final_state[j].set_index('Run Number',append=True,inplace=True)
        final_state[j] = final_state[j].reorder_levels(['Run Number',0,1])
        final_state[j].index.names=[None,None,None]
    
    #PREPARE ROW FOR METADATA TABLE
    metadata = pd.DataFrame([K,q,e,fs,fw,Ddiv,M,S,food,richness],columns=[run_number],
                             index=['K','q','e','fs','fw','Ddiv','M','S','Food','Rich']).T

    #STAMP CONSUMER MATRIX WITH RUN NUMBER
    c_matrix = pd.DataFrame(params['c'].copy(),columns=MyPlate.R.index,index=MyPlate.N.index)
    c_matrix['Run Number']=run_number
    c_matrix.set_index('Run Number',append=True,inplace=True)
    c_matrix = c_matrix.reorder_levels(['Run Number',0,1])
    c_matrix.index.names=[None,None,None]
    
    return final_state + [metadata,c_matrix,params,Ntraj,Rtraj]
