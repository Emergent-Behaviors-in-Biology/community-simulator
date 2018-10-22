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

def RunCommunity(K=1000.,q=0.,e=0.4,fs=0.25,fw=0.25,food_type=0,food_type_2=1,mix_frac=1,Ddiv=0.2,n_types=4,c1=1,
                 c0=0.01,muc=10,MA=25,SA=40,Sgen=40,S=100,n_iter=200,T=5,n_wells=27,run_number=0,
                 params=None,N0=None,R0=None,extra_time=False,sample_kind='Binary',
                 sigm=0.1,sigw=0,sige=0,scale=1e6):
    
    MA = int(round(MA))
    SA = int(round(SA))
    Sgen = int(round(Sgen))
    S=int(round(S))

    if sample_kind is not 'Binary':
        print('Only binary sampling is implemented!')
        return 'Error'

    sample_par = {'SA': SA*np.ones(n_types+1), #Number of species in each family
          'MA': np.asarray(np.hstack((10*np.ones(n_types),[MA])),dtype=int), #Number of resources of each type
          'Sgen': Sgen, #Number of generalist species
          'muc': muc, #Mean sum of consumption rates,
          'q': q, #Preference strength 
          'c0':c0, #Background consumption rate in binary model
          'c1':c1, #Maximum consumption rate in binary model
          'fs':fs, #Fraction of secretion flux with same resource type
          'fw':fw, #Fraction of secretion flux to 'waste' resource
          'D_diversity':Ddiv #Variability in secretion fluxes among resources (must be less than 1)
         }

    #Create resource vector and set food supply
    M = int(np.sum(sample_par['MA']))
    if R0 is None:
        R0 = np.zeros((M,n_wells))
        R0[food_type,:] = K*mix_frac
        R0[food_type_2,:] = K*(1-mix_frac)

    #Create initial conditions (sub-sampling from regional species pool)
    S_tot = int(np.sum(sample_par['SA']))+sample_par['Sgen']
    if N0 is None:
        N0 = np.zeros((S_tot,n_wells))
        for k in range(n_wells):
            N0[np.random.choice(S_tot,size=S,replace=False),k]=1.

    #Create parameter set
    if params is None:
        if sample_kind == 'Gaussian':
            sample_par['muc'] = muc + c0
        
        c, D = usertools.MakeMatrices(params=sample_par, kind=sample_kind, waste_ind=n_types)
        params={'c':c,
                'm':np.ones(S_tot)+sigm*np.random.randn(S_tot),
                'w':np.ones(M)+sigw*np.random.randn(M),
                'D':D,
                'g':np.ones(S_tot),
                'e':e+sige*np.random.randn(M),
                'r':1.,
                'tau':1,
                'K':20
                }
    else:
        params['e'] = e
        
    params['R0']=np.zeros(M)
    params['R0'][food_type] = K*mix_frac
    params['R0'][food_type_2] = K*(1-mix_frac)
        
    N0,R0 = usertools.AddLabels(N0,R0,params['c'])
    init_state = [N0,R0]
    

    MyPlate = Community(init_state,dynamics,params)
    
    try:
        Ntraj,Rtraj = MyPlate.RunExperiment(np.eye(n_wells),T,n_iter,refresh_resource=False,scale=scale)
        if extra_time:
            Ntraj2,Rtraj2 = MyPlate.RunExperiment(np.eye(n_wells),100,10,refresh_resource=False,scale=scale)
            Ntraj2,Rtraj2 = MyPlate.RunExperiment(np.eye(n_wells),1000,10,refresh_resource=False,scale=scale)
        MyPlate.Passage(np.eye(n_wells),refresh_resource=False,scale=scale)
    except:
        Ntraj = np.nan
        Rtraj = np.nan
        MyPlate.N = MyPlate.N*np.nan
        print('Run failed with the following sample parameters: ')
        print(sample_par)
    richness = np.mean((MyPlate.N>0).sum().values)

    
    final_state = [MyPlate.N.copy(), MyPlate.R.copy()]
    for j in range(2):
        final_state[j]['Run Number']=run_number
        final_state[j].set_index('Run Number',append=True,inplace=True)
        final_state[j] = final_state[j].reorder_levels(['Run Number',0,1])
        final_state[j].index.names=[None,None,None]
    
    params_in = pd.DataFrame([K,q,e,muc,c1,c0,fs,fw,Ddiv,M,S,food_type,food_type_2,mix_frac,richness,sigm,sigw,sige,sample_kind],columns=[run_number],
                             index=['K','q','e','muc','c1','c0','fs','fw','Ddiv','M','S','Food','Food2','mix','Rich','sig_m','sig_w','sig_e','c_sampling']).T

    c_matrix = pd.DataFrame(params['c'].copy(),columns=MyPlate.R.index,index=MyPlate.N.index)
    c_matrix['Run Number']=run_number
    c_matrix.set_index('Run Number',append=True,inplace=True)
    c_matrix = c_matrix.reorder_levels(['Run Number',0,1])
    c_matrix.index.names=[None,None,None]
    
    return final_state + [params_in,c_matrix,params,Ntraj,Rtraj]
