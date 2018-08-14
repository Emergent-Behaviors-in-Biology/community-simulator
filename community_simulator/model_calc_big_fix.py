#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:23:10 2017

@author: robertmarsland
"""

import argparse
from community_simulator.model_study import RunCommunity
from community_simulator.model_study_II import RunCommunity_II
import numpy as np
import distutils.dir_util
import pandas as pd
import datetime
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("task_ID", type=int)
parser.add_argument("date", type=str)
args = parser.parse_args()

#SET UP FILE NAMES
#folder = 'test'
folder = '/project/biophys/microbial_crm/data'
distutils.dir_util.mkpath(folder)
datanames = ['Consumers','Resources','Parameters','Initial_State','Realization']
suff = ['.xlsx']*4+['.dat']
ic = [[0,1,2],[0,1,2],0]
h = [0,0,0]
filenames = [folder+'/'+datanames[j]+'_'+args.date+'_'+str(args.task_ID)+suff[j] for j in range(5)]

#ITERATIONS, ETC.
n_iter = 100
trials = 10
ns = 10
T=5

#LOAD OLD FILES
N0 = pd.read_excel(filenames[3],index_col=ic[3],header=h[3])
with open(filenames_old[4],'rb') as f:
    params = pickle.load(f)

#SYSTEM SIZE
MA = 125
SA = 200
n_types = 4
M = MA*n_types

#CHOOSE PARAMETERS
Kvec = (0.01*M)*(10**np.linspace(1,3,ns))
evec = np.linspace(0.1,1,ns)
kwargs ={'K':Kvec[-1],
         'e':evec[0],
         'run_number':0,
         'n_iter':n_iter,
         'T':T,
         'n_wells':trials,
         'extra_time':True,
         'MA':MA,
         'SA':SA,
         'Sgen':SA,
         'n_types':n_types,
         'scale':1e9,
         'S':M,
         'c0':1./M,
         'N0':N0.values,
         'params':params
         }

out = RunCommunity(**kwargs)
        
for q in range(3):
    old = pd.read_excel(filenames[q],index_col=ic[q],header=h[q])
    old.append(out[q]).to_excel(filenames[q])
