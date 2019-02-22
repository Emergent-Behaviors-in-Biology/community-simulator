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

R0_food = 1000

mp = {'sampling':'Binary', #Sampling method
    'SA': 5000, #Number of species in each family
    'MA': 300, #Number of resources of each type
    'Sgen': 0, #Number of generalist species
    'muc': 10, #Mean sum of consumption rates in Gaussian model
    'q': 0, #Preference strength (0 for generalist and 1 for specialist)
    'c0':0, #Background consumption rate in binary model
    'c1':1., #Specific consumption rate in binary model
    'fs':0.45, #Fraction of secretion flux with same resource type
    'fw':0.45, #Fraction of secretion flux to 'waste' resource
    'D_diversity':0.3, #Variability in secretion fluxes among resources (must be less than 1)
    'regulation':'independent',
    'replenishment':'external',
    'response':'type I'
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
food_list = np.asarray([item for k in range(16) for item in [k]*125])
n_samples = len(food_list)
EMP_protocol = {'R0_food':R0_food, #unperturbed fixed point for supplied food
                'n_wells':n_samples, #Number of independent wells
                'S':2500, #Number of species per well
                'food':food_list #index of food source
                }
EMP_protocol.update(mp)

#Make initial state
N0,R0 = AddLabels(*MakeInitialState(EMP_protocol),c)
init_state=[N0,R0]
metadata = pd.DataFrame(food_list,index=N0.T.index,columns=['Food Source'])

#Make parameter list
m = 0.1+0.01*np.random.randn(len(c))
params=[{'w':1,
        'g':1,
        'l':0.8,
        'R0':R0.values[:,k],
        'm':m+4.5*np.random.rand(),
        'tau':1
        } for k in range(len(N0.T))]
for k in range(len(params)):
    params[k]['c'] = c
    params[k]['D'] = D

metadata['m'] = np.asarray([np.mean(item['m']) for item in params])
EMP = Community(init_state,dynamics,params)
EMP.SteadyState(verbose=True,plot=False,tol=1e-3,eps=0.1)
with open('/project/biophys/microbial_crm/data/EMP.dat','wb') as f:
    pickle.dump([EMP.N,EMP.R,params[0],R0,metadata],f)

for k in range(len(params)):
    params[k]['m'] = m + food_list[k]*4.5/15
metadata['m'] = np.asarray([np.mean(item['m']) for item in params])
EMP = Community(init_state,dynamics,params)
EMP.SteadyState(verbose=True,plot=False,tol=1e-3)
with open('/project/biophys/microbial_crm/data/EMP_corr.dat','wb') as f:
    pickle.dump([EMP.N,EMP.R,params[0],R0,metadata],f)

