#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:09:38 2017

@author: robertmarsland
"""
import numpy as np
import pandas as pd
import pexpect
import os

username = 'marsland'
hostname = 'scc1.bu.edu'
directory = '/project/biophys/microbial_crm'

def rsync_in(source,dest,username=username,hostname=hostname,directory=directory,password=None):
    fullsource = '/'.join([directory,source])
    fullhost = '@'.join([username,hostname])
    command = ' '.join(['rsync -r -avz',':'.join([fullhost,fullsource]),dest])
    child = pexpect.spawn(command)
    child.expect('Password:')
    child.sendline(password)
    child.expect('speedup')
    print(child.before)

def FormatPath(folder):
    if folder==None:
        folder=''
    else:
        if folder != '':
            if folder[-1] != '/':
                folder = folder+'/'
    return folder

def LoadData(folder,date,load_all=False):
    folder = FormatPath(folder)
    N = pd.read_excel(folder+'Consumers_'+date+'.xlsx',index_col=[0,1,2],header=[0])
    R = pd.read_excel(folder+'Resources_'+date+'.xlsx',index_col=[0,1,2],header=[0])
    c = pd.read_excel(folder+'c_matrix_'+date+'.xlsx',index_col=[0,1,2],header=[0,1])
    params = pd.read_excel(folder+'Parameters_'+date+'.xlsx',index_col=[0],header=[0])
    
    if load_all:
        with open(folder+'Realization_'+date+'.dat','rb') as f:
            full_params = pickle.load(f)
        return N,R,c,params,full_params
    else:
        return N,R,c,params

def ComputeIPR(df):
    IPR = pd.DataFrame(columns=df.keys(),index=df.index.levels[0])
    for j in df.index.levels[0]:
        p = df.loc[j]/df.loc[j].sum()
        IPR.loc[j] = 1./((p[p>0]**2).sum())
    return IPR

def PostProcess(folder,date,cutoff=0):
    folder = FormatPath(folder)
    N,R,c,params = LoadData(folder,date=date)
    N_IPR = ComputeIPR(N)
    R_IPR = ComputeIPR(R)
    
    ns = len(N_IPR)*len(N_IPR.keys())
    
    data = pd.DataFrame(index=list(range(ns)),columns=['Plate','Consumer IPR','Resource IPR']+list(params.keys()))
    
    j=0
    
    for plate in N_IPR.index:
        for well in N_IPR.keys():
            data.loc[j,'Plate'] = plate
            data.loc[j,'Consumer IPR'] = N_IPR.loc[plate,well]
            data.loc[j,'Resource IPR'] = R_IPR.loc[plate,well]
            for item in params.keys():
                data.loc[j,item] = params.loc[plate,item]
            data.loc[j,'Consumer Richness']=(N.loc[plate]>cutoff*(N.loc[plate].sum())).sum()[well]
            data.loc[j,'Resource Richness']=(R.loc[plate]>cutoff*(R.loc[plate].sum())).sum()[well]
            j+=1
    data.to_excel(folder+'data_'+date+'.xlsx')
    return data

def FlatResult(N,R,params):
    types = R.index.levels[1]
    n_wells = len(N.keys())
    Nflat = N.loc[N.index.levels[0][0]].T
    Nflat.index = np.arange(n_wells)
    metadata = pd.DataFrame()
    metadata['Community'] = np.arange(n_wells)
    metadata['Food'] = params['Food'].loc[N.index.levels[0][0]]
    
    k=1
    for rn in N.index.levels[0][1:]:
        Nflat_temp = N.loc[N.index.levels[0][rn]].T
        Nflat_temp.index = np.arange(n_wells)+k*n_wells
        metadata_temp = pd.DataFrame()
        metadata_temp['Community'] = np.arange(n_wells)
        metadata_temp['Food'] = params['Food'].loc[rn]
        metadata_temp.index = Nflat_temp.index
        Nflat = Nflat.append(Nflat_temp)
        metadata = metadata.append(metadata_temp)
        k+=1
    metadata['Food Type'] = types[R.index.labels[1][metadata['Food']]]
    return Nflat, metadata
