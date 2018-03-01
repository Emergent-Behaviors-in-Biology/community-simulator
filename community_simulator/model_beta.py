#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 17:23:10 2017

@author: robertmarsland
"""

import argparse
from community_simulator.model_study import RunCommunity
import numpy as np
import distutils.dir_util
import pandas as pd
import datetime
import pickle

#SET UP FILE NAMES
#folder = 'test'
folder = '/project/biophys/microbial_crm/data'
distutils.dir_util.mkpath(folder)
datanames = ['Consumers','Resources','Parameters','Initial_State','Realization']
suff = ['.xlsx']*4+['.dat']
ic = [[0,1,2],[0,1,2],0]
h = [0,0,0]
filenames = [folder+'/'+datanames[j]+'_'+str(datetime.datetime.now()).split()[0]+'_'+str(args.task_ID)+suff[j] for j in range(5)]

#ITERATIONS, ETC.
n_iter = 100
trials = 999
T=5

#CHOOSE PARAMETERS
Kvec = [28,1000,1000]
evec = [0.4,0.9,0.1]
kwargs ={'K':Kvec[0],
        'e':evec[0],
        'run_number':0,
        'n_iter':n_iter,
        'T':T,
        'n_wells':trials,
        'extra_time':True
        }

#LOOP THROUGH PARAMETERS
first_run = True
for j in range(len(Kvec)):
    if not first_run:
        kwargs['run_number'] = j
        kwargs['e'] = evec[j]
        kwargs['K'] = Kvec[j]
        
    #RUN COMMUNITY
    out = RunCommunity(**kwargs)
        
    #ON FIRST RUN, SAVE PARAMETERS AND INITIAL CONDITION
    if first_run:
        params = out[4]
        with open(filenames[4],'wb') as f:
            pickle.dump(params,f)
        for q in range(3):
            out[q].to_excel(filenames[q])
        N0 = out[5].loc[0].T
        N0.to_excel(filenames[3])
        kwargs.update({'params':params,'N0':N0.values})
    #ON SUBSEQUENT RUNS, APPEND NEW RESULTS TO FILES
    else:
        for q in range(3):
            old = pd.read_excel(filenames[q],index_col=ic[q],header=h[q])
            old.append(out[q]).to_excel(filenames[q])
    del out
    first_run = False
