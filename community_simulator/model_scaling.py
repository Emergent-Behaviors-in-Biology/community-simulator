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
parser.add_argument("param", type=str)
parser.add_argument("min", type=float)
parser.add_argument("max", type=float)
parser.add_argument("ns", type=int)
parser.add_argument("extra_param", type=str)
parser.add_argument("extra_param_val", type=float)
parser.add_argument("n_iter", type=int)
parser.add_argument("ind_trials", type=int)
args = parser.parse_args()

valid_params=['K' ,'q', 'e', 'fs', 'fw', 'food_type', 'Ddiv', 'n_types', 'c1','c0', 'MA', 'SA', 'Sgen', 'S', 'n_wells','sigm']
assert args.param in valid_params, 'Invalid choice of variable parameter.'
assert args.extra_param in valid_params+['None'], 'Invalid choice of extra parameter.'

folder = 'test'
#folder = '/project/biophys/microbial_crm/'+args.param+'data/'+args.extra_param+str(args.extra_param_val)
distutils.dir_util.mkpath(folder)
datanames = ['Consumers','Resources','Parameters','c_matrix','Realization']
ic = [[0,1,2],[0,1,2],0,[0,1,2]]
h = [0,0,0,[0,1]]
filenames = [folder+'/'+datanames[j]+'_'+str(datetime.datetime.now()).split()[0]+'.xlsx' for j in range(4)]
filenames.append(folder+'/'+datanames[4]+'_'+str(datetime.datetime.now()).split()[0]+'.dat')

T=5

if args.extra_param != 'None':
    extra_params = {args.extra_param:args.extra_param_val}
else:
    extra_params = {}
if args.extra_param != 'n_wells':
    extra_params['n_wells'] = 10

paramvec=np.linspace(args.min,args.max,args.ns)
f = open(filenames[4],'wb')
for j in range(args.ns):
    for k in range(args.ind_trials):
        print(args.param+'='+str(paramvec[j]))
        kwargs = {args.param:paramvec[j],'run_number':j*args.ind_trials+k,'n_iter':args.n_iter,'T':T}
        kwargs.update(extra_params)
        out = RunCommunity(**kwargs)
        pickle.dump([out[4]],f)
        if j==0 and k==0:
            for q in range(4):
                out[q].to_excel(filenames[q])
        else:
            for q in range(4):
                old = pd.read_excel(filenames[q],index_col=ic[q],header=h[q])
                old.append(out[q]).to_excel(filenames[q])
        del out
f.close()
