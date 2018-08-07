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
date = str(datetime.datetime.now()).split()[0]
filenames = [folder+'/'+datanames[j]+'_'+date+'_'+str(args.task_ID)+suff[j] for j in range(5)]

#ITERATIONS, ETC.
n_iter = 100
trials = 10
ind_trials = 10
T=5

#CHOOSE PARAMETERS
MAvec = [10,20,30,40,50,60,70,80,90,100,110,120,130,140]
n_types = 4
MA = MAvec[args.task_ID-6]
M = MA*n_types
S = M
Stot = S*2
SA = Stot/(n_types+1)
Sgen = SA

#Kvec = np.ones(10)*10*M
#evec = np.linspace(0.1,1,10)
#Kvec = np.asarray([0.46,2.15,10.0])*M
#evec = np.asarray([0.5,0.5,0.5])
Kvec = np.asarray([0.1,0.28,10.0,10.0])*M
evec = np.asarray([0.1,0.4,0.9,0.1])
#Kvec = (10**np.linspace(1,3,ns))*M/100

kwargs ={'K':Kvec[0],
        'e':evec[0],
        'run_number':0,
        'n_iter':n_iter,
        'T':T,
        'n_wells':trials,
        'extra_time':True,
        'n_types':n_types,
        'SA':SA,
        'MA':MA,
        'S':S,
        'Sgen':Sgen,
        'scale':1e9,
        'c0':1./M
        }

#LOOP THROUGH PARAMETERS
first_run = True
    
for j in range(len(Kvec)):
    print('K='+str(Kvec[j]))
    print('l='+str(1-evec[j]))
    
    if not first_run:
        kwargs['run_number'] = j
        kwargs['K'] = Kvec[j]
        kwargs['e'] = evec[j]
    
    for k in range(ind_trials):
        
        kwargs['run_number'] = j*ind_trials + k

        #RUN COMMUNITY
        out = RunCommunity(**kwargs)
        
        if first_run:
            for q in range(3):
                out[q].to_excel(filenames[q])
        else:
            for q in range(3):
                old = pd.read_excel(filenames[q],index_col=ic[q],header=h[q])
                old.append(out[q]).to_excel(filenames[q])
        del out
        first_run = False
