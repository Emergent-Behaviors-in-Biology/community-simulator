#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 14:32:09 2017

@author: robertmarsland
"""

import argparse
from community_simulator.cavity_threelevel import RunCommunity, cost_function_bounded
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import scipy.optimize as opt
import distutils.dir_util

parser = argparse.ArgumentParser()
parser.add_argument("task_ID", type=int)
parser.add_argument("param", type=str)
args = parser.parse_args()

param_names = ['K','sigK','muc','sigc','mud','sigd','m','sigm','u','sigu','gamma','eta']
arg_names = ['<R>','<N>','<X>','<R^2>','<N^2>','<X^2>']

folder = '/project/biophys/trophic_structure/dataDec17/vary_'+args.param
folder_new = '/project/biophys/trophic_structure/dataDec17/vary_'+args.param+'/new_opt'
distutils.dir_util.mkpath(folder_new)

filename = folder+'/'+'data_'+str(args.task_ID)+'_K_eta'+'.xlsx' 
filename_new = folder_new+'/'+'data_'+str(args.task_ID)+'_K_eta'+'.xlsx' 
data = pd.read_excel(filename,index_col=0)
data_new = data.copy()
interp_data = data.copy()
good_data = data[data['fun']<0.0001]
for item in arg_names:
    interp_data[item] = griddata(good_data[['K','eta']],good_data[item],data[['K','eta']])

for item in data.index:
    params = data[param_names].loc[item].to_dict()
    args = data[arg_names].loc[item].values
    if np.all(np.isfinite(interp_data[arg_names].loc[item].values)):
        out = opt.minimize(cost_function_bounded,interp_data[arg_names].loc[item].values,args=(params,))
    else:
        out = opt.minimize(cost_function_bounded,data[arg_names].loc[item].values,args=(params,))
    if np.isfinite(out.fun):
        data_new.loc[item,arg_names] = out.x
        data_new.loc[item,'fun'] = out.fun
        
data_new.to_excel(filename_new)


filename = folder+'/'+'data_'+str(args.task_ID)+'_muc_mud'+'.xlsx' 
filename_new = folder_new+'/'+'data_'+str(args.task_ID)+'_muc_mud'+'.xlsx' 
data = pd.read_excel(filename,index_col=0)
data_new = data.copy()

for item in data.index:
    params = data[param_names].loc[item].to_dict()
    args = data[arg_names].loc[item].values
    out = opt.minimize(cost_function_bounded,data[arg_names].loc[item],args=(params,))
    if np.isfinite(out.fun):
        data_new.loc[item,arg_names] = out.x
        data_new.loc[item,'fun'] = out.fun
        
data_new.to_excel(filename_new)