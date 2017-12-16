#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:32:09 2017

@author: robertmarsland
"""

import argparse
from community_simulator.cavity_threelevel import RunCommunity
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("task_ID", type=int)
args = parser.parse_args()

folder = '/project/biophys/trophic_structure/dataDec17/'

params = {'K':1.,
          'sigK':0.1,
          'muc':1.,
          'sigc':0.1+0.01*(args.task_ID-1),
          'mud':1.,
          'sigd':0.1,
          'm':0.5,
          'sigm':0.01,
          'u':0.1,
          'sigu':0.01,
          'gamma':1.,
          'eta':1.}
S = 50

Kvec = np.linspace(0,1.5,3)
for j in range(len(Kvec)):
    params['K']=Kvec[j]
    data_new, final_state_new, sim_params_new, c_matrix_new= RunCommunity(params,S,trials=12,run_number=j,n_iter=100)
    
    if j==0:
        final_state = final_state_new.copy()
        data = data_new.copy()
        sim_params = sim_params_new.copy()
        #c_matrix = c_matrix_new.copy()
    else:
        final_state = final_state.append(final_state_new.copy())
        data = data.append(data_new.copy())
        sim_params = sim_params.append(sim_params_new.copy())
        #c_matrix = c_matrix.append(c_matrix_new.copy())
        
namelist = ['finalstate','data','simparams','cmatrix']
j=0
for item in [final_state,data,sim_params]:
    item.to_csv(folder+namelist[j]+'_'+str(args.task_ID)+'.csv')
    j+=1
