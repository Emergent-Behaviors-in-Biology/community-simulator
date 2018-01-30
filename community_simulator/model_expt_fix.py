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
parser.add_argument("date", type=str)
parser.add_argument("fs", type=float)
parser.add_argument("fw", type=float)
parser.add_argument("q", type=float)
parser.add_argument("n_iter", type=int)
parser.add_argument("start", type=int)
args = parser.parse_args()

#folder = 'test'
folder = '/project/biophys/microbial_crm/'+args.foldername
distutils.dir_util.mkpath(folder)
datanames = ['Consumers','Resources','Parameters','c_matrix','Realization','Initial_State']
ic = [[0,1,2],[0,1,2],0,[0,1,2]]
h = [0,0,0,[0,1]]
filenames = [folder+'/'+datanames[j]+'_'+args.date+'.xlsx' for j in range(4)]
filenames = filenames +[folder+'/'+datanames[j]+'_'+args.date+'.dat'
                        for j in [4,5]]
trials = 10
T=5
MA = 7
n_types = 3
M=MA*n_types

with open(filenames[4],'rb') as f:
    params = pickle.load(f)[0]
with open(filenames[5],'rb') as f:
    N0 = pickle.load(f)[0]

for k in range(args.start,M):
    kwargs = {'food_type':k,'run_number':j*M+k,'n_iter':args.n_iter,'T':T,'c1':2,'SA':50,'Sgen':0,'S':20,
              'n_wells':trials,'MA':MA,'q':args.q,'fw':args.fw,'fs':args.fs,'n_types':n_types}
    kwargs.update({'params':params,'N0':N0.values})

    out = RunCommunity(**kwargs)
        
    for q in range(4):
        old = pd.read_excel(filenames[q],index_col=ic[q],header=h[q])
        old.append(out[q]).to_excel(filenames[q])


