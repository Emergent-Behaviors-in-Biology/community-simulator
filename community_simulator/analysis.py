#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:09:38 2017

@author: robertmarsland
"""
import numpy as np
import pandas as pd
import pexpect

def rsync_in(source,dest,username,hostname,directory,password):
    fullsource = '/'.join([directory,source])
    fullhost = '@'.join([username,hostname])
    command = ' '.join(['rsync -r -avz',':'.join([fullhost,fullsource]),dest])
    child = pexpect.spawn(command)
    child.expect('Password:')
    child.sendline(password)
    child.expect('speedup')
    print(child.before)

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

def MakeFlux(assumptions):
    """
    Make function to compute matrix of incoming fluxes as a function of resource
    abundance and parameter dictionary
    """
    
    sigma = {'type I': lambda R,params: params['c']*R,
             'type II': lambda R,params: params['c']*R/(1+params['c']*R/params['K']),
             'type III': lambda R,params: params['c']*(R**params['n'])/(1+params['c']*(R**params['n'])/params['K'])
            }
    
    u = {'independent': lambda x,params: 1.,
         'energy': lambda x,params: (((params['w']*x)**params['nreg']).T
                                      /np.sum((params['w']*x)**params['nreg'],axis=1)).T,
         'mass': lambda x,params: ((x**params['nreg']).T/np.sum(x**params['nreg'],axis=1)).T
        }
    
    J_in = lambda R,params: (u[assumptions['regulation']](params['c']*R,params)
                            *params['w']*sigma[assumptions['response']](R,params))
    return J_in
                                                
def Susceptibility(N,R,beta,params):
    """
    Compute partial derivatives of steady-state populations N and resource
    abundances R with respect to the supplied energy flux kappa_beta = R0_beta/tau_beta
    """
    
    if type(params['c']) is pd.DataFrame:
        c = params['c'].values
    else:
        c = params['c']
    if type(params['D']) is pd.DataFrame:
        D = params['D'].values
    else:
        D = params['D']
    M = len(D)
    l = params['l']
    tau = params['tau']
    
    not_extinct = np.where(N > 0)[0]
    c = c[not_extinct,:]
    S_star = len(not_extinct)
    N = N[not_extinct]

    A1 = (D*l-np.eye(M))*(c.T.dot(N))-np.eye(M)/tau
    A2 = (D*l-np.eye(M)).dot((c*R).T)
    A3 = np.hstack(((1-l)*c,np.zeros((S_star,S_star))))
    A = np.vstack((np.hstack((A1,A2)),A3))
    b = np.zeros(M+S_star)
    b[beta] = -1/tau

    Ainv = np.linalg.inv(A)
    chieta = Ainv.dot(b)

    chi = chieta[:M] #Resource susceptibilities
    eta = chieta[M:] #Species susceptibilities
    
    return chi, eta

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

def LotkaVolterra(N,R,params):
    """
    Compute effective Lotka-Volterra coefficients and carrying capacity for 
    dynamics near the fixed point.
    """
    
    if type(params['c']) is pd.DataFrame:
        c = params['c'].values
    else:
        c = params['c']
    if type(params['D']) is pd.DataFrame:
        D = params['D'].values
    else:
        D = params['D']
    M = len(D)
    l = params['l']
    w = params['w']
    tau = params['tau']
    M = len(params['D'])
    
    
    A = -(1/tau+ c.T.dot(N))*np.eye(M) + ((D*((c*l*w).T.dot(N))).T/w).T
    Q = (c*R).T - ((D.dot((c*R*w*l).T)).T/w).T
    dRdN = np.linalg.inv(A).dot(Q)
    
    alpha = -(c*w*(1-l)).dot(dRdN)
    K = alpha.dot(N)
    
    return K, alpha

def validate_simulation(com_in,N0):
    """
    Check accuracy, convergence, and noninvadability of community instance com_in.
    N0 indicates which species were present at the beginning of the simulation.
    """
    
    com = com_in.copy()
    failures = np.sum(np.isnan(com.N.iloc[0]))
    survive = com.N>0
    com.N[survive] = 1
    if type(com.params) is not list:
        params_list = [com.params for k in range(len(com.N.T))]
    else:
        params_list = com.params
    dlogNdt_survive = pd.DataFrame(np.asarray(list(map(com.dNdt,com.N.T.values,com.R.T.values,params_list))).T,
                                   index=com.N.index,columns=com.N.columns)

    com.N[N0>0] = 1
    com.N[survive] = 0
    dlogNdt_extinct = pd.DataFrame(np.asarray(list(map(com.dNdt,com.N.T.values,com.R.T.values,params_list))).T,
                               index=com.N.index,columns=com.N.columns)

    accuracy = np.max(abs(dlogNdt_survive))
    invaders = np.sum(dlogNdt_extinct>0)
    
    return {'Mean Accuracy':accuracy.mean(),'Std. Dev. Accuracy':accuracy.std(),'Failures':failures,'Invasions':(invaders>0).sum()}
