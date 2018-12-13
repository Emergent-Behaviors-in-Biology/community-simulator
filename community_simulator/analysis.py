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
import pickle

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

def MakeFlux(response='type I',regulation='independent'):

    sigma = {'type I': lambda R,params: params['c'].values*R,
             'type II': lambda R,params: params['c'].values*R/(1+params['c'].values*R/params['K']),
             'type III': lambda R,params: params['c'].values*(R**params['n'])/(1+params['c'].values*(R**params['n'])/params['K'])
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

def CalculateConsumptionMeff(N,R,par):
    M_eff = []
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

def NODF(A):
    """
        Compute the Nestdness metric based on Overlap and Decreasing Fill (NODF) as
        defined in Almeida-Neto et al. (2008).
    """
    
    m,n = np.shape(A)
    Ac = np.ones((n,n))*A.sum(axis=0)
    Ar = np.ones((m,m))*A.T.sum(axis=0)
    Dr = Ar<Ar.T
    Dc = Ac<Ac.T
    B = ((A.T)/(A.T.sum(axis=0))).T
    C = A/A.sum(axis=0)
    
    return 2*(np.trace(A.T.dot(Dr.dot(B)))+np.trace(A.dot(Dc.dot(C.T))))/(n*(n-1)+m*(m-1))

def LotkaVolterra(N,R,par):
    M = len(par['D'])
    par['e'] = np.ones(M)*par['e']
    par['l'] = 1- par['e']
    
    c = par['c'].values
    
    A = -(1/par['tau'] + c.T.dot(N))*np.eye(M) + ((par['D']*((c*par['l']*par['w']).T.dot(N))).T/par['w']).T
    Q = (c*R).T - ((par['D'].dot((c*R*par['w']*par['l']).T)).T/par['w']).T
    dRdN = np.linalg.inv(A).dot(Q)
    
    alpha = -(c*par['w']*par['e']).dot(dRdN)
    K = alpha.dot(N)
    
    return K, alpha

