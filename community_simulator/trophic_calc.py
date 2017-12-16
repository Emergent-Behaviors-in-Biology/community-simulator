#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:32:09 2017

@author: robertmarsland
"""

import argparse
from community_simulator.cavity_threelevel import RunCommunity
import numpy as np
import distutils.dir_util

parser = argparse.ArgumentParser()
parser.add_argument("task_ID", type=int)
parser.add_argument("param", type=str)
parser.add_argument("scale", type=double)
parser.add_argument("ns", type=int)
args = parser.parse_args()

folder = '/project/biophys/trophic_structure/dataDec17/vary_'+args.param
distutils.dir_util.mkpath(folder)

params = {'K':1.,
          'sigK':0.,
          'muc':1.,
          'sigc':0.1,
          'mud':1.,
          'sigd':0.1,
          'm':0.5,
          'sigm':0.01,
          'u':0.5,
          'sigu':0.01,
          'gamma':1.,
          'eta':1.}

params[args.param] = args.scale*args.task_ID

S = 50

Kvec = np.linspace(0,1.5,args.ns)
etavec = np.linspace(0.3,5,args.ns)
for j in range(len(Kvec)):
    for m in range(len(etavec)):
        params['K']=Kvec[j]
        params['eta']=etavec[m]
        data_new, final_state_new, sim_params_new, c_matrix_new= RunCommunity(params,S,trials=36,run_number=j,n_iter=800)
    
        if j==0 and k==0:
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
    item.to_csv(folder+'/'+namelist[j]+'_'+str(args.task_ID)+'_K_eta'+'.csv')
    j+=1

mucvec = np.linspace(0.5,2.5,args.ns)
mudvec = np.linspace(0.5,2,args.ns)
for j in range(len(mucvec)):
    for m in range(len(mudvec)):
        params['muc']=mucvec[j]
        params['mud']=mudvec[m]
        data_new, final_state_new, sim_params_new, c_matrix_new= RunCommunity(params,S,trials=36,run_number=j,n_iter=800)
        
        if j==0 and k==0:
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
    item.to_csv(folder+'/'+namelist[j]+'_'+str(args.task_ID)+'_muc_mud'+'.csv')
    j+=1
