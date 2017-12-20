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
assumptions = {'regulation':'independent','replenishment':'non-renew','response':'type I'}
def dNdt(N,R,params):
    return usertools.MakeConsumerDynamics(**assumptions)(N,R,params)
def dRdt(N,R,params):
    return usertools.MakeResourceDynamics(**assumptions)(N,R,params)
dynamics = [dNdt,dRdt]

def RunCommunity(productivity=10.,q=0.2,e=0.5,MA=10,S=50,n_iter=1000,T=5,
                 n_wells=27,run_number=0):
    
    sample_par = {'SA': 20*np.ones(4), #Number of species in each family
          'MA': MA*np.ones(4), #Number of resources of each type
          'Sgen': 20, #Number of generalist species
          'muc': 10, #Mean sum of consumption rates in Gaussian model
          'sigc': .01, #Variance in consumption rate in Gaussian model
          'q': q, #Preference strength 
          'c0':0.01, #Background consumption rate in binary model
          'c1':0.5, #Maximum consumption rate in binary model
          'fs':0.2, #Fraction of secretion flux with same resource type
          'fw':0.7, #Fraction of secretion flux to 'waste' resource
          'D_diversity':0.2 #Variability in secretion fluxes among resources (must be less than 1)
         }

    c, D = usertools.MakeMatrices(params=sample_par, kind='Binary', waste_ind=3)

    #Create initial conditions (sub-sampling from regional species pool)
    S_tot = len(c)
    M = len(D)
    N0 = np.zeros((S_tot,n_wells))
    for k in range(n_wells):
        N0[np.random.choice(S_tot,size=S,replace=False),k]=1e-3/S
    R0 = np.zeros((M,n_wells))
    R0[0,:] = productivity

    N0,R0 = usertools.AddLabels(N0,R0,c)
    init_state = [N0,R0]

    #Create parameter set
    params={'c':c,
            'm':np.ones(S_tot),
            'w':np.ones(M),
            'D':D,
            'g':np.ones(S_tot),
            'e':e,
            'R0':R0.values[:,0],
            'r':1.,
            'tau':1
            }

    MyPlate = Community(init_state,dynamics,params)

    MyPlate.RunExperiment(np.eye(n_wells),T,n_iter,refresh_resource=False,scale=1e6)
    final_state = [MyPlate.N, MyPlate.R]
    for j in range(2):
        final_state[j]['Run Number']=run_number
        final_state[j].set_index('Run Number',append=True,inplace=True)
        final_state[j] = final_state[j].reorder_levels(['Run Number',0,1])
        final_state[j].index.names=[None,None,None]
        
    params_in = pd.DataFrame([productivity,q,e,MA,S],columns=[run_number],
                             index=['productivity','q','e','MA','S']).T
        
    return final_state + [params_in]