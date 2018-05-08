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
args = parser.parse_args()

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
n_iter = 500
trials = 10
ns = 10
T=0.1

#CHOOSE PARAMETERS
Kvec = 10**np.linspace(1,3,ns)
evec = np.linspace(0.1,1,ns)
kwargs ={'K':Kvec[0],
        'e':evec[0],
        'run_number':0,
        'n_iter':n_iter,
        'T':T,
        'n_wells':trials,
        'extra_time':False
        }
#kwargs_list = [{},{'Ddiv':0.001},{'sample_kind':'Gaussian'},{'sige':0.03,'sigw':0.1}]
kwargs_list = [{},{'sample_kind':'Gaussian'}]
kwargs.update(kwargs_list[args.task_ID - 1])

#LOOP THROUGH PARAMETERS
first_run = True
for j in range(len(Kvec)):
    print('K='+str(Kvec[j]))
    for m in range(len(evec)):
        #FOR SUBSEQUENT RUNS, KEEP OLD MATRICES AND INITIAL CONDITIONS
        if not first_run:
            kwargs['run_number'] = j*len(evec)+m
            kwargs['e'] = evec[m]
            kwargs['K'] = Kvec[j]
        
        #RUN COMMUNITY
        if args.task_ID == 1:
        	out = RunCommunity_II(**kwargs)
        else:
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
