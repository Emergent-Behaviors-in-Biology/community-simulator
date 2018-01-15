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
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("task_ID", type=int)
parser.add_argument("param", type=str)
args = parser.parse_args()

param_names = ['K','sigK','muc','sigc','mud','sigd','m','sigm','u','sigu','gamma','eta']
folder = '/project/biophys/trophic_structure/dataDec17_rev/vary_'+args.param
distutils.dir_util.mkpath(folder_new)
namelist = ['data','finalstate','simparams','cmatrix']
filenamelist = [folder+'/'+namelist[q]+'_'+str(args.task_ID)+'_K_eta'+'.xlsx' for q in range(len(namelist))]
ic = [0,[0,1,2],0,[0,1]]
h = [0,0,[0,1],[0,1]]

old=[]
for q in range(4):
    old.append(pd.read_excel(filenamelist[q],index_col=ic[q],header=h[q]))

n_iter = 1000
trials = 27
T=5
cutoff = 1e-6
S = 40

for item in data.index:
    params = old[0][param_names].loc[item].to_dict()
    if params['K']>=0.7:
        out = RunCommunity(params,S,trials=trials,run_number=item,
                           n_iter=n_iter,T=T,cutoff=cutoff)
        
        for q in range(4):
            old[q].loc[item]=out[q]
            old[q].to_excel(filenamelist[q])

        del out