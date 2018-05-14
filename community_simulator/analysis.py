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

def LoadData(folder,date,load_all=False,load_c=True,task_id=None):
    folder = FormatPath(folder)
    if task_id == None:
        task_id = ''
    else:
        task_id = '_'+str(task_id)

    N = pd.read_excel(folder+'Consumers_'+date+task_id+'.xlsx',index_col=[0,1,2],header=[0])
    R = pd.read_excel(folder+'Resources_'+date+task_id+'.xlsx',index_col=[0,1,2],header=[0])
    params = pd.read_excel(folder+'Parameters_'+date+task_id+'.xlsx',index_col=[0],header=[0])
    
    if load_c:
        c = pd.read_excel(folder+'c_matrix_'+date+'.xlsx',index_col=[0,1,2],header=[0,1])
        return N,R,c,params
    if load_all:
        c = pd.read_excel(folder+'c_matrix_'+date+'.xlsx',index_col=[0,1,2],header=[0,1])
        with open(folder+'Realization_'+date+'.dat','rb') as f:
            full_params = pickle.load(f)
        return N,R,c,params,full_params
    else:
        return N,R,params

def FlatResult(N,R,params):
    types = R.index.levels[1]
    n_wells = len(N.keys())
    Nflat = N.loc[N.index.levels[0][0]].T
    Nflat.index = np.arange(n_wells)
    Rflat = R.loc[R.index.levels[0][0]].T
    Rflat.index = np.arange(n_wells)
    metadata = pd.DataFrame()
    metadata['Community'] = np.arange(n_wells)
    for item in params:
        metadata[item] = params[item].loc[N.index.levels[0][0]]
    metadata.index = np.arange(n_wells)
    
    k=1
    for rn in N.index.levels[0][1:]:
        Nflat_temp = N.loc[N.index.levels[0][rn]].T
        Rflat_temp = R.loc[N.index.levels[0][rn]].T
        Nflat_temp.index = np.arange(n_wells)+k*n_wells
        Rflat_temp.index = np.arange(n_wells)+k*n_wells
        metadata_temp = pd.DataFrame()
        metadata_temp['Community'] = np.arange(n_wells)
        for item in params:
            metadata_temp[item] = params[item].loc[rn]
        metadata_temp.index = Nflat_temp.index
        Nflat = Nflat.append(Nflat_temp)
        Rflat = Rflat.append(Rflat_temp)
        metadata = metadata.append(metadata_temp)
        k+=1
    metadata['Food Type'] = types[R.index.labels[1][metadata['Food']]]
    return Nflat, Rflat, metadata

def Simpson(N):
    p = N/np.sum(N)
    return 1./np.sum(p**2)

def Shannon(N):
    p = N/np.sum(N)
    p = p[p>0]
    return np.exp(-np.sum(p*np.log(p)))

def BergerParker(N):
    p = N/np.sum(N)
    return 1./np.max(p)

def Richness(N,thresh=0):
    return np.sum(N>thresh)

metrics = {'Simpson':Simpson,'Shannon':Shannon,'BergerParker':BergerParker,'Richness':Richness}

def CalculateDiversity(df,metadata):
    metadata_new = metadata.copy()
    for function in metrics:
        metadata_new[function]=np.nan
        for item in df.index:
            metadata_new.loc[item,function]=metrics[function](df.loc[item].values)
    return metadata_new

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

def MakeFlux(response='type I',regulation='independent'):

    sigma = {'type I': lambda R,params: params['c']*R,
             'type II': lambda R,params: params['c']*R/(1+params['c']*R/params['K']),
             'type III': lambda R,params: params['c']*(R**params['n'])/(1+params['c']*(R**params['n'])/params['K'])
            }
    
    u = {'independent': lambda x,params: 1.,
         'energy': lambda x,params: (((params['w']*x)**params['nreg']).T
                                      /np.sum((params['w']*x)**params['nreg'],axis=1)).T,
         'mass': lambda x,params: ((x**params['nreg']).T/np.sum(x**params['nreg'],axis=1)).T
        }
    
    J_in = lambda R,params: (u[regulation](params['c']*R,params)
                             *params['w']*sigma[response](R,params))
    
    return J_in

Jin = MakeFlux()

def CalculateConsumptionMeff(N,R,par_in):
    M_eff = []
    par = par_in.copy()
    par['c'] = par['c'].values
    par['D'] = par['D'].values
    for well in N.keys():
        N1 = N[well].values
        R1 = R[well].values
        not_extinct = np.where(N1 > 0)[0]
        for species in not_extinct:
            M_eff.append(Simpson(Jin(R1,par)[species,:]))
    return M_eff

def CalculateConsumptionNeff(N,c,metric='Simpson',thresh=2):
    cap_vec = []
    if metric == 'Simpson':
        for well in N.keys():
            N1 = N[well].values
            c_red = c[N1>0,:]
            cap_1 = np.asarray([Simpson(c_red[:,k]) for k in range(np.shape(c)[1])])
            cap_1[np.where(np.sum(c_red,axis=0)<thresh*np.mean(c.reshape(-1)))] = 0
            cap_vec = cap_vec + list(cap_1[1:])

    else:
        c_norm = c-np.min(c.reshape(-1))
        c_norm = c_norm/np.max(c_norm.reshape(-1))
        for well in N.keys():
            N1 = N[well].values
            cap_1 = c_norm.T.dot(N1>0)
            cap_vec = cap_vec + list(cap_1[1:])

    
    return np.asarray(cap_vec)

def Susceptibility(N1,R1,beta,par):
    R1 = np.asarray(R1,dtype=float)
    N1 = np.asarray(N1,dtype=float)
    M = len(par['D'].values)
    l = 1 - par['e']
    not_extinct = np.where(N1 > 0)[0]

    c = par['c'].values[not_extinct,:]
    Sphi = len(not_extinct)
    N1 = N1[not_extinct]

    A1 = (par['D'].values*l-np.eye(M))*(c.T.dot(N1))-np.eye(M)/par['tau']
    A2 = (par['D'].values*l-np.eye(M)).dot((c*R1).T)
    A3 = np.hstack(((1-l)*c,np.zeros((Sphi,Sphi))))
    A = np.vstack((np.hstack((A1,A2)),A3))
    b = np.zeros(M+Sphi)
    b[beta] = -1/par['tau']

    Ainv = np.linalg.inv(A)
    chieta = Ainv.dot(b)

    chi = chieta[:M]
    eta = chieta[M:]
    
    return chi, eta

def CalculateSusceptibility(N,R,par,print_progress=False):
    chi_diag = []
    chi_off = []
    for well in N.keys():
        N1 = N[well].values
        R1 = R[well].values
        for beta in range(len(R1)):
            chi, eta = Susceptibility(N1,R1,beta,par)
            if beta > 0:
                chi_diag.append(chi[beta])
            chi = list(chi)
            del chi[beta]
            chi_off = chi_off + chi
        if print_progress:
            print(well+' done.')
    
    chi_diag = np.asarray(chi_diag)
    chi_off = np.asarray(chi_off)
    
    return chi_diag, chi_off



