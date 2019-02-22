#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:23:10 2017

@author: robertmarsland
"""
import pandas as pd
import numpy as np
from community_simulator.usertools import MakeConsumerDynamics,MakeResourceDynamics,MakeMatrices,MakeInitialState,AddLabels
from community_simulator import Community
import pickle

n_samples = 300
R0_food = 1000

mp = {'sampling':'Binary', #Sampling method
    'SA': np.ones(6)*800, #Number of species in each family
    'MA': np.ones(6)*50, #Number of resources of each type
    'Sgen': 200, #Number of generalist species
    'muc': 10, #Mean sum of consumption rates in Gaussian model
    'q': 0.9, #Preference strength (0 for generalist and 1 for specialist)
    'c0':0, #Background consumption rate in binary model
    'c1':1., #Specific consumption rate in binary model
    'fs':0.45, #Fraction of secretion flux with same resource type
    'fw':0.45, #Fraction of secretion flux to 'waste' resource
    'D_diversity':0.3, #Variability in secretion fluxes among resources (must be less than 1)
    'regulation':'independent',
    'replenishment':'external',
    'response':'type I',
    'waste_type':5
    }

#Construct dynamics
def dNdt(N,R,params):
    return MakeConsumerDynamics(mp)(N,R,params)
def dRdt(N,R,params):
    return MakeResourceDynamics(mp)(N,R,params)
dynamics = [dNdt,dRdt]

#Construct matrices
c,D = MakeMatrices(mp)

#Set up the experiment
#NOTE: R0_food and food are irrelevant here, because we make a new
#R0 matrix later on
HMP_protocol = {'R0_food':R0_food, #unperturbed fixed point for supplied food
                'n_wells':3*n_samples, #Number of independent wells
                'S':2500, #Number of species per well
                'food':0 #index of food source
                }
HMP_protocol.update(mp)

#Make initial state
N0,R0 = MakeInitialState(HMP_protocol)
R0 = np.zeros(np.shape(R0))
alpha = np.linspace(0,1,n_samples)
for k in range(3):
    R0[2*k*50,k*n_samples:(k+1)*n_samples] = alpha*R0_food
    R0[(2*k+1)*50,k*n_samples:(k+1)*n_samples] = (1-alpha)*R0_food
N0,R0 = AddLabels(N0,R0,c)
init_state=[N0,R0]

#Make parameter list
m = 1+0.01*np.random.randn(len(c))
params=[{'w':1,
        'g':1,
        'l':0.8,
        'R0':R0.values[:,k],
        'r':1.,
        'tau':1
        } for k in range(len(N0.T))]
for k in range(len(params)):
    params[k]['c'] = c
    params[k]['D'] = D
    params[k]['m'] = m

HMP = Community(init_state,dynamics,params)
HMP.metadata = pd.DataFrame(['Env. 1']*n_samples+['Env. 2']*n_samples+['Env. 3']*n_samples,
                            index=N0.T.index,columns=['Environment'])

HMP.SteadyState(verbose=True,plot=False,tol=1e-3)

with open('/project/biophys/microbial_crm/data/HMP_env_family.dat','wb') as f:
    pickle.dump([HMP.N,HMP.R,params[0],R0,HMP.metadata],f)
