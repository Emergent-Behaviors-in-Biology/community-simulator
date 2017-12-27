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

def RunCommunity(K=10.,q=0.,e=0.5,MA=25,S=100,n_iter=200,T=5,
                 n_wells=27,run_number=0,fs=0.25,fw=0.25):
    
    sample_par = {'SA': 40*np.ones(4), #Number of species in each family
          'MA': MA*np.ones(4), #Number of resources of each type
          'Sgen': 40, #Number of generalist species
          'muc': 10, #Mean sum of consumption rates
          'q': q, #Preference strength 
          'c0':0.01, #Background consumption rate in binary model
          'c1':1., #Maximum consumption rate in binary model
          'fs':fs, #Fraction of secretion flux with same resource type
          'fw':fw, #Fraction of secretion flux to 'waste' resource
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
    R0[0,:] = K

    N0,R0 = usertools.AddLabels(N0,R0,c)
    init_state = [N0,R0]

    #Create parameter set
    params={'c':c,
            'm':np.ones(S_tot)*0.1,
            'w':np.ones(M),
            'D':D,
            'g':np.ones(S_tot),
            'e':e,
            'R0':R0.values[:,0],
            'r':1.,
            'tau':1
            }

    MyPlate = Community(init_state,dynamics,params)
    
    try:
        MyPlate.RunExperiment(np.eye(n_wells),T,n_iter,refresh_resource=False,scale=1e6)
        MyPlate.Passage(np.eye(n_wells),refresh_resource=False)
        richness = np.mean((MyPlate.N>0).sum().values)
    except:
        richness = np.nan
    
    final_state = [MyPlate.N.copy(), MyPlate.R.copy()]
    for j in range(2):
        final_state[j]['Run Number']=run_number
        final_state[j].set_index('Run Number',append=True,inplace=True)
        final_state[j] = final_state[j].reorder_levels(['Run Number',0,1])
        final_state[j].index.names=[None,None,None]
        
    params_in = pd.DataFrame([K,q,e,MA,S,richness],columns=[run_number],
                             index=['K','q','e','MA','S','Rich']).T

    c_matrix = pd.DataFrame(c,columns=MyPlate.R.index,index=MyPlate.N.index)
    c_matrix['Run Number']=run_number
    c_matrix.set_index('Run Number',append=True,inplace=True)
    c_matrix = c_matrix.reorder_levels(['Run Number',0,1])
    c_matrix.index.names=[None,None,None]
        
    return final_state + [params_in,c_matrix]
