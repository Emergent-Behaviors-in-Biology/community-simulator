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
parser.add_argument("foldername", type=str)
parser.add_argument("fs", type=float)
parser.add_argument("fw", type=float)
parser.add_argument("q", type=float)
parser.add_argument("e", type=float)
parser.add_argument("K", type=float)
parser.add_argument("n_iter", type=int)
parser.add_argument("ind_trials", type=int)
args = parser.parse_args()

#folder = 'test'
folder = '/project/biophys/microbial_crm/'+args.foldername
distutils.dir_util.mkpath(folder)
datanames = ['Consumers','Resources','Parameters','c_matrix','Realization','Initial_State']
ic = [[0,1,2],[0,1,2],0,[0,1,2]]
h = [0,0,0,[0,1]]
filenames = [folder+'/'+datanames[j]+'_'+str(datetime.datetime.now()).split()[0]+'.xlsx' for j in range(4)]
filenames = filenames +[folder+'/'+datanames[j]+'_'+str(datetime.datetime.now()).split()[0]+'.dat'
                        for j in [4,5]]
trials = 10
T=5
MA = 25
n_types = 4
M=MA*n_types

for j in range(args.ind_trials):
    first_run = True
    
    for k in range(M):
        kwargs = {'food_type':k,'run_number':j*M+k,'n_iter':args.n_iter,'T':T,'c1':1,'SA':50,'Sgen':0,'S':20,
                  'n_wells':trials,'MA':MA,'q':args.q,'fw':args.fw,'fs':args.fs,'n_types':n_types,'e':args.e,
                  'K':args.K,'extra_time':True}
        if not first_run:
            kwargs.update({'params':params,'N0':N0.values})

        out = RunCommunity(**kwargs)
        params = out[4]
        N0 = out[5].loc[0].T
        
        if first_run and j==0:
            for q in range(4):
                out[q].to_excel(filenames[q])
            with open(filenames[4],'wb') as f:
                pickle.dump([out[4]],f)
            with open(filenames[5],'wb') as f:
                pickle.dump([N0],f)
        else:
            for q in range(4):
                old = pd.read_excel(filenames[q],index_col=ic[q],header=h[q])
                old.append(out[q]).to_excel(filenames[q])
            if first_run:
                with open(filenames[4],'rb') as f:
                    paramlist = pickle.load(f)
                with open(filenames[4],'wb') as f:
                    pickle.dump(paramlist+[out[4]],f)
                del paramlist

                with open(filenames[5],'rb') as f:
                    N0list = pickle.load(f)
                with open(filenames[5],'wb') as f:
                    pickle.dump(N0list+[N0],f)
                del N0list

        first_run = False


