    #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:51:13 2017

@author: robertmarsland
"""

import numpy as np
from scipy.stats import norm
from community_simulator import Community,essentialtools,usertools,visualization,analysis
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import seaborn as sns
import scipy.optimize as opt
import pandas as pd

#############################
#SET UP PROBLEM
#############################

#Moments of truncated Gaussian
def w0(Delta):
    return norm.cdf(Delta)
def w1(Delta):
    return Delta + norm.pdf(-Delta)-Delta*norm.cdf(-Delta)
def w2(Delta):
    return (Delta**2+1)*(1-norm.cdf(-Delta))+Delta*norm.pdf(-Delta)
def y(Delta):
    return w2(Delta)/(w1(Delta)**2)

#Standard deviations of invasion growth rates
def sigN(args,params):
    R,N,qR = args
    return np.sqrt(params['sigm']**2 + (1-params['l'])**2*params['sigc']**2
                  *(params['mug']**2+params['sigg']**2)*qR)
def sigd(args,params):
    R,N,qR = args
    qN = y(DelN(args,params))*N**2
    return np.sqrt(params['sigw']**2 + params['sigc']**2*qN/params['gamma'])
def sigp(args,params):
    R,N,qR = args
    qN = y(DelN(args,params))*N**2
    return np.sqrt(params['l']**2*(params['sigc']**2*params['sigD']**2*qR*qN+
                                   params['muc']**2*params['sigD']**2*N**2*qR/params['gamma'])
                                   /params['gamma'])

#Mean invasion growth rates normalized by standard deviation
def DelN(args,params):
    R,N,qR = args
    return ((1-params['l'])*params['muc']*params['mug']*R-params['m'])/sigN(args,params)

#Fraction of species that survive in steady state
def phiN(args,params):
    return w0(DelN(args,params))

#Susceptibilities
def nu(args,params):
    R,N,qR = args
    omega_eff = params['omega']+params['muc']*N/params['gamma']
    kappa_eff = params['kappa']+params['l']*params['muc']*N*R/params['gamma']
    r2 = kappa_eff*phiN(args,params)/(params['gamma']*omega_eff*R)
    nu0 = (2*params['gamma']**(-2)*phiN(args,params)**2*kappa_eff/R**2)*(1+np.sqrt(1+(2*r2)**(-2)))
    r1 = kappa_eff*nu0/omega_eff**2
    eps_pk = sigp(args,params)**2/kappa_eff**2
    eps_d = sigd(args,params)**2/omega_eff**2
    return nu0*(1-(6*r1**2*eps_pk-(2*r1-1)*eps_d)/(6*r1**3*r2**(-2)+r1**2*r2**(-2)-16*r1**2-8*r1))

def chi(args,params):
    R,N,qR = args
    omega_eff = params['omega']+params['muc']*N/params['gamma']
    kappa_eff = params['kappa']+params['l']*params['muc']*N*R/params['gamma']
    r2 = kappa_eff*phiN(args,params)/(params['gamma']*omega_eff*R)
    nu0 = (2*params['gamma']**(-2)*phiN(args,params)**2*kappa_eff/R**2)*(1+np.sqrt(1+(2*r2)**(-2)))
    chi0 = (R*params['gamma']/(2*phiN(args,params)*kappa_eff))/(1+np.sqrt(1+(2*r2)**(-2)))
    r1 = kappa_eff*nu0/omega_eff**2
    eps_pk = sigp(args,params)**2/kappa_eff**2
    eps_d = sigd(args,params)**2/omega_eff**2
    return chi0*(1+(6*r1**2*eps_pk-(2*r1-1)*eps_d)/(6*r1**3*r2**(-2)+r1**2*r2**(-2)-16*r1**2-8*r1))

#Factors for converting invasion growth rate to steady-state abundance
def fN(args,params):
    R,N,qR = args
    return ((1-params['l'])*params['mug']*chi(args,params)*params['sigc']**2*R)**(-1)

#Test satisfaction of competitive exclusion bound for consumers and predators
def test_bound_1(args,params):
    return params['gamma']-phiN(args,params)

def cost_vector(args,params):
    R,N,qR = args
    omega_eff = params['omega']+params['muc']*N/params['gamma']
    kappa_eff = params['kappa']+params['l']*params['muc']*N*R/params['gamma']
    nubar = nu(args,params)*(1-params['l'])*params['mug']*params['sigc']**2/params['gamma']
    r1 = kappa_eff*nubar/omega_eff**2
    eps_pw = sigp(args,params)**2*nubar**2/omega_eff**4
    eps_d = sigd(args,params)**2/omega_eff**2

    RHS = np.asarray([(omega_eff/(2*nubar))*(np.sqrt(1+4*r1)-1-2*(eps_pw-r1*eps_d)/(1+4*r1)**(3/2)),
                     fN(args,params)*sigN(args,params)*w1(DelN(args,params)),
                     (omega_eff**2/(2*nubar**2))*(1+2*r1-np.sqrt(1+4*r1)+eps_d*(1-(1+6*r1)/((1+4*r1)**(3/2)))+2*eps_pw/(1+4*r1)**(3/2))])
    
    return RHS

def cost_vector_single(args,params):
    R,qR = args
    N = params['N']
    args = np.asarray([R,N,qR])
    omega_eff = params['omega']+params['muc']*N/params['gamma']
    kappa_eff = params['kappa']+params['l']*params['muc']*N*R/params['gamma']
    nubar = nu(args,params)*(1-params['l'])*params['mug']*params['sigc']**2/params['gamma']
    r1 = kappa_eff*nubar/omega_eff**2
    eps_pw = sigp(args,params)**2*nubar**2/omega_eff**4
    eps_d = sigd(args,params)**2/omega_eff**2

    RHS = np.asarray([(omega_eff/(2*nubar))*(np.sqrt(1+4*r1)-1-2*(eps_pw-r1*eps_d)/(1+4*r1)**(3/2)),
                     (omega_eff**2/(2*nubar**2))*(1+2*r1-np.sqrt(1+4*r1)+eps_d*(1-(1+6*r1)/((1+4*r1)**(3/2)))+2*eps_pw/(1+4*r1)**(3/2))])
    
    return RHS

#Return sum of squared differences between RHS and LHS of self-consistency eqns
def cost_function(args,params):
    args = np.exp(args)
    return np.sum((args-cost_vector(args,params))**2)

def cost_function_single(args,params):
    args = np.exp(args)
    return np.sum((args-cost_vector_single(args,params))**2)

#Enforce competitive exclusion bounds and keep moments within reasonable values
# def cost_function_bounded(args,params):
#     b1 = test_bound_1(args,params)
#     if np.isfinite(b1):
#         if b1>0 and np.all(args>=0):
#             return cost_function(args,params)
#         else:
#             return np.inf
#     else:
#         return np.inf

#############################
#SOLVE PROBLEM
#############################
    
#Generate dynamics for McArthur CRM with predator
assumptions = {
          'regulation':'independent', #metabolic regulation (see dRdt)
          'response':'type I', #functional response (see dRdt)
          'supply':'external' #resource supply (see dRdt)
         }

def dNdt(N,R,params):
    return usertools.MakeConsumerDynamics(assumptions)(N,R,params)
def dRdt(N,R,params):
    return usertools.MakeResourceDynamics(assumptions)(N,R,params)

#Run community to steady state, extract moments of steady state, plot results
def RunCommunity(assumptions,M,eps=1e-5,trials=1,postprocess=True,
                 run_number=0,cutoff=1e-5,max_iter=1):
    
    fun = np.inf
    k=0
    
    assumptions['n_wells'] = trials
    assumptions['waste_type'] = 0
    assumptions['MA'] = M
    assumptions['Sgen'] = 0
    S = int(round(M/assumptions['gamma']))
    Stot = S*2
    assumptions['S'] = S
    assumptions['SA'] = Stot
    if assumptions['sampling'] == 'Binary':
        p_c = 1/((M*assumptions['sigc']**2/assumptions['muc']**2)+1)
        assumptions['c1'] = assumptions['muc']/(M*p_c)
    assumptions['omega'] = 1/assumptions['tau']
    assumptions['sigw'] = 0
    assumptions['sigD'] = np.sqrt((assumptions['sparsity']/(assumptions['sparsity']+1))*((M-1)/M))
    assumptions['mug'] = 1
    assumptions['sigg'] = 0
    
    if assumptions['single']:
        #Get closed-form solution for large <N> limit of single
        args_closed = np.asarray([np.nan,np.nan,np.nan])
        y0 = assumptions['muc']**2*((1/assumptions['l'])-1)/(assumptions['gamma']*assumptions['sigc']**2)
        out = minimize_scalar(lambda Delta:(y(Delta)-y0)**2,bracket=[-5,5])
        #r2 = lambda Delta: assumptions['l']*w0(Delta)/assumptions['gamma']
        #y0 = lambda Delta: (assumptions['muc']**2/(assumptions['gamma']*assumptions['sigc']**2))*((1/assumptions['l'])-(1-r2(Delta)))*(1+4*r2(Delta))**(3/2)
        #y0 = lambda Delta: (assumptions['muc']**2/(assumptions['gamma']*assumptions['sigc']**2))*(r2(Delta)*assumptions['sigD']**2/(assumptions['l']-assumptions['sigD']**2) + ((1/assumptions['l'])-(1-r2(Delta)))*(1+4*r2(Delta))**(3/2))/(1-r2(Delta)*assumptions['sigD']**2/(1-r2(Delta)*assumptions['sigD']**2/(assumptions['l']-assumptions['sigD']**2)))
        #out = minimize_scalar(lambda Delta:(y(Delta)-y0(Delta))**2,bracket=[-5,5])

        DelN_closed = out.x
        R0 = assumptions['m']/((1-assumptions['l'])*assumptions['mug']*assumptions['muc'])
        r3 = assumptions['sigc']*DelN_closed/(assumptions['muc']*np.sqrt(assumptions['l']-assumptions['sigD']**2))
        r4 = assumptions['sigm']*DelN_closed/assumptions['m']
        R_closed = R0*(1+r3*np.sqrt(1-r4**2))/(1-r3**2)
        qR_closed = (R_closed**2)/(assumptions['l']-assumptions['sigD']**2)
        args_closed[0] = R_closed
        args_closed[2] = qR_closed
        args_closed[1] = assumptions['R0']/(S*assumptions['m'])
        assumptions['kappa'] = 0
    else:
        assumptions['kappa'] = np.mean(assumptions['R0']/assumptions['tau'])
    
    while fun > eps and k < max_iter:
        params = usertools.MakeParams(assumptions)
        params['R0'] = assumptions['R0']
        if assumptions['single']:
            params['R0'] = np.zeros(M)
            params['R0'][0] = assumptions['R0']
        N0,R0 = usertools.MakeInitialState(assumptions)
    
        TestPlate = Community([N0,R0],[dNdt,dRdt],params)
        TestPlate.SteadyState()
    
        #Find final states
        TestPlate.N[TestPlate.N<cutoff] = 0
        Rmean = TestPlate.R.mean(axis=0)
        R2mean = (TestPlate.R**2).mean(axis=0)
        Nmean = (Stot*1./S)*TestPlate.N.mean(axis=0)
    
        #Compute moments for feeding in to cavity calculation
        args0 = np.asarray([np.mean(Rmean),
                            np.mean(Nmean),
                            np.mean(R2mean)])+1e-10
        args0_err = np.asarray([np.std(Rmean),
                                np.std(Nmean),
                                np.std(R2mean)])
        
        if assumptions['single']:
            bounds = [(np.log(args0[k])-1,np.log(args0[k])+1) for k in [0,2]]
            assumptions['N'] = args0[1]
            out = opt.minimize(cost_function_single,np.log(args0[[0,2]]),args=(assumptions,),bounds=bounds,tol=1e-8)
            args_cav = np.exp(out.x)
            args_cav = [args_cav[0],args0[1],args_cav[1]]
        else:
            bounds = [(np.log(args_closed[k])-1,np.log(args_closed[k])+1) for k in range(3)]
            out = opt.minimize(cost_function,np.log(args_closed),args=(assumptions,),bounds=bounds,tol=1e-8)
            #ranges = [(np.log(args_closed[k])-1,np.log(args_closed[k])+1) for k in range(4)]
            #out = opt.brute(cost_function,ranges,Ns=100,args=(assumptions,),workers=-1)
            args_cav = np.exp(out.x)
        fun = out.fun#/np.sum(args0**2)
        k += 1
    if fun > eps:
        args_cav = [np.nan,np.nan,np.nan]
        fun = np.nan
    
    if postprocess:
        results_num = {'SphiN':np.sum(TestPlate.N.values.reshape(-1)>cutoff)*1./trials,
                       'M<R>':M*args0[0],
                       'S<N>':S*args0[1],
                       'MqR':M*args0[2]}
        results_cav = {'SphiN':S*phiN(args_cav,assumptions),
                       'M<R>':M*args_cav[0],
                       'S<N>':S*args_cav[1],
                       'MqR':M*args_cav[2]}
        if assumptions['single']:
            results_closed = {'SphiN':S*w0(DelN_closed),
                            'M<R>':M*args_closed[0],
                            'S<N>':S*args_closed[1],
                            'MqR':M*args_closed[2]}
        else:
            results_closed = np.nan
        return results_num, results_cav, results_closed, out, args0, assumptions, TestPlate

    else:

        data = pd.DataFrame([args_cav],columns=['<R>','<N>','<R^2>'],index=[run_number])
        data['fun']=fun
        for item in assumptions.keys():
            data[item]=assumptions[item]
        data['S'] = S
        data['M'] = M
        data['phiN'] = phiN(args_cav,assumptions)
        data['<N^2>'] = y(DelN(args_cav,assumptions))*args_cav[1]**2

        data_sim = pd.DataFrame([args0],columns=['<R>','<N>','<R^2>'],index=[run_number])
        data_sim['phiN'] = np.mean((TestPlate.N>cutoff).sum(axis=0))/S
        data_sim['<N^2>'] = np.mean((Stot*1./S)*(TestPlate.N**2).mean(axis=0))
        err_sim = pd.DataFrame([args0_err],columns=['<R>','<N>','<R^2>'],index=[run_number])
        err_sim['phiN'] = np.std((TestPlate.N>cutoff).sum(axis=0))/S
        err_sim['<N^2>'] = np.std((Stot*1./S)*(TestPlate.N**2).mean(axis=0))
        data = data.join(data_sim,rsuffix='_sim').join(err_sim,rsuffix='_sim_err')

        if assumptions['single']:
            data_closed = pd.DataFrame([args_closed],columns=['<R>','<N>','<R^2>'],index=[run_number])
            data_closed['phiN'] = w0(DelN_closed)
            data_closed['<N^2>'] = y(DelN_closed)*args_closed[1]**2
            data = data.join(data_closed,rsuffix='_closed')

        return data
