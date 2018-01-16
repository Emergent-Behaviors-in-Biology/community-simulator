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

def PostProcess(folder,date):
    folder = FormatPath(folder)
    N,R,c,params = LoadData(folder,date=date)
    N_IPR = ComputeIPR(N)
    R_IPR = ComputeIPR(R)
    
    ns = len(N_IPR)*len(N_IPR.keys())
    
    data = pd.DataFrame(index=list(range(ns)),columns=['Plate','Consumer IPR', 'Resource IPR']+list(params.keys()))
    
    j=0
    
    for plate in N_IPR.index:
        for well in N_IPR.keys():
            data.loc[j,'Plate'] = plate
            data.loc[j,'Consumer IPR'] = N_IPR.loc[plate,well]
            data.loc[j,'Resource IPR'] = R_IPR.loc[plate,well]
            for item in params.keys():
                data.loc[j,item] = params.loc[plate,item]
            data.loc[j,'Rich']=(N.loc[plate]>0).sum()[well]
            j+=1
    data.to_excel(folder+'data_'+date+'.xlsx')
    return data