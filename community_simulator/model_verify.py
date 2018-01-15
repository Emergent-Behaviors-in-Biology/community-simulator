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
parser.add_argument("ns", type=int)
args = parser.parse_args()

#folder = 'test'
folder = '/project/biophys/microbial_crm/verify'
distutils.dir_util.mkpath(folder)
datanames = ['Consumers','Resources','Parameters','c_matrix','Realization']
ic = [[0,1,2],[0,1,2],0,[0,1,2]]
h = [0,0,0,[0,1]]
filenames = [folder+'/'+datanames[j]+'_'+str(datetime.datetime.now()).split()[0]+'.xlsx' for j in range(4)]

n_iter = 50
trials = 27
T=5

for j in range(args.ns):
    out = RunCommunity(q=0,run_number=j,n_iter=n_iter,T=T,n_wells=trials,SA=50,Sgen=0,fw=0.7,fs=0.25)
        
    if j==0:
        for q in range(4):
            out[q].to_excel(filenames[q])
        with open(filenames[4],'wb') as f:
            pickle.dump([out[4]],f)
    else:
        for q in range(4):
            old = pd.read_excel(filenames[q],index_col=ic[q],header=h[q])
            old.append(out[q]).to_excel(filenames[q])
        with open(filenames[4],'rb') as f:
            paramlist = pickle.load(f)
        with open(filenames[4],'wb') as f:
            pickle.dump(paramlist+[out[4]],f)
    del out
