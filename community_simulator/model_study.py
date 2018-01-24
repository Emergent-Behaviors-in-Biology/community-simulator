#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:25:50 2017

@author: robertmarsland
"""

from community_simulator import Community,usertools
import numpy as np
import pandas as pd

#Construct dynamics
assumptions = {'regulation':'independent','replenishment':'renew','response':'type I'}
def dNdt(N,R,params):
    return usertools.MakeConsumerDynamics(**assumptions)(N,R,params)
def dRdt(N,R,params):
    return usertools.MakeResourceDynamics(**assumptions)(N,R,params)
dynamics = [dNdt,dRdt]

def RunCommunity(K=500.,q=0.,e=0.2,fs=0.25,fw=0.25,food_type=0,Ddiv=0.2,n_types=4,c1=1,
                 MA=25,SA=40,Sgen=40,S=100,n_iter=200,T=5,n_wells=27,run_number=0,
                 params=None,N0=None):
    
    MA = int(round(MA))
    SA = int(round(SA))
    Sgen = int(round(Sgen))
    S=int(round(S))
    
    sample_par = {'SA': SA*np.ones(n_types), #Number of species in each family
          'MA': MA*np.ones(n_types), #Number of resources of each type
          'Sgen': Sgen, #Number of generalist species
          'muc': 10, #Mean sum of consumption rates
          'q': q, #Preference strength 
          'c0':0.01, #Background consumption rate in binary model
          'c1':c1, #Maximum consumption rate in binary model
          'fs':fs, #Fraction of secretion flux with same resource type
          'fw':fw, #Fraction of secretion flux to 'waste' resource
          'D_diversity':0.2 #Variability in secretion fluxes among resources (must be less than 1)
         }

    #Create resource vector and set food supply
    M = int(np.sum(sample_par['MA']))
    R0 = np.zeros((M,n_wells))
    R0[food_type,:] = K

    #Create initial conditions (sub-sampling from regional species pool)
    S_tot = int(np.sum(sample_par['SA']))+sample_par['Sgen']
    if N0 is None:
        N0 = np.zeros((S_tot,n_wells))
        for k in range(n_wells):
            N0[np.random.choice(S_tot,size=S,replace=False),k]=1e-3/S

    #Create parameter set
    if params is None:
        c, D = usertools.MakeMatrices(params=sample_par, kind='Binary', waste_ind=n_types-1)
        params={'c':c,
                'm':np.ones(S_tot)*0.5+np.random.rand(S_tot),
                'w':np.ones(M),
                'D':D,
                'g':np.ones(S_tot),
                'e':e,
                'r':1.,
                'tau':1
                }
    N0,R0 = usertools.AddLabels(N0,R0,params['c'])
    init_state = [N0,R0]
    params['R0']=R0.values[:,0]
    MyPlate = Community(init_state,dynamics,params)
    
    try:
        Ntraj,Rtraj = MyPlate.RunExperiment(np.eye(n_wells),T,n_iter,refresh_resource=False,scale=1e6)
        MyPlate.Passage(np.eye(n_wells),refresh_resource=False)
        richness = np.mean((MyPlate.N>0).sum().values)
    except:
        richness = np.nan
        Ntraj = np.nan
        Rtraj = np.nan
    
    final_state = [MyPlate.N.copy(), MyPlate.R.copy()]
    for j in range(2):
        final_state[j]['Run Number']=run_number
        final_state[j].set_index('Run Number',append=True,inplace=True)
        final_state[j] = final_state[j].reorder_levels(['Run Number',0,1])
        final_state[j].index.names=[None,None,None]
        
    params_in = pd.DataFrame([K,q,e,fs,fw,Ddiv,M,S,food_type,richness],columns=[run_number],
                             index=['K','q','e','fs','fw','Ddiv','M','S','Food','Rich']).T

    c_matrix = pd.DataFrame(params['c'].copy(),columns=MyPlate.R.index,index=MyPlate.N.index)
    c_matrix['Run Number']=run_number
    c_matrix.set_index('Run Number',append=True,inplace=True)
    c_matrix = c_matrix.reorder_levels(['Run Number',0,1])
    c_matrix.index.names=[None,None,None]
    
    return final_state + [params_in,c_matrix,params,Ntraj,Rtraj]
