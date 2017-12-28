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

parser = argparse.ArgumentParser()
#parser.add_argument("task_ID", type=int)
#parser.add_argument("param", type=str)
#parser.add_argument("scale", type=float)
parser.add_argument("ns", type=int)
args = parser.parse_args()

#folder = 'test'
folder = '/project/biophys/microbial_crm/data'
distutils.dir_util.mkpath(folder)
datanames = ['Consumers','Resources','Parameters','c_matrix']
ic = [[0,1,2],[0,1,2],0,[0,1,2]]
h = [0,0,0,[0,1]]
filenames = [folder+'/'+datanames[j]+'_'+str(datetime.datetime.now()).split()[0]+'.xlsx' for j in range(4)]

n_iter = 500
trials = 27
T=5

Kvec = np.linspace(10,1000,args.ns)
evec = np.linspace(0.1,1,args.ns)
for j in range(len(Kvec)):
    print('K='+str(Kvec[j]))
    for m in range(len(evec)):
        out = RunCommunity(K=Kvec[j],e=evec[m],run_number=j*len(evec)+m,
                           n_iter=n_iter,T=T,n_wells=trials)
        
        if j==0 and m==0:
            for q in range(4):
                out[q].to_excel(filenames[q])
        else:
            for q in range(4):
                old = pd.read_excel(filenames[q],index_col=ic[q],header=h[q])
                old.append(out[q]).to_excel(filenames[q])
        del out
