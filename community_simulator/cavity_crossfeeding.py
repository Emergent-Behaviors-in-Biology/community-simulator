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

#Standard deviations of invasion growth rates
def sigN(args,params):
    R,N,qR,qN = args
    return np.sqrt(params['sigm']**2 + (1-params['l'])**2*params['sigc']**2
                  *(params['mug']**2+params['sigg']**2)*qR)
def sigd(args,params):
    R,N,qR,qN = args
    return np.sqrt(params['sigw']**2 + params['sigc']**2*qN/params['gamma'])
def sigp(args,params):
    R,N,qR,qN = args
    return np.sqrt(params['l']**2*(params['sigc']**2*params['sigD']**2*qR*qN+
                                   params['muc']**2*params['sigD']**2*N**2*qR/params['gamma'])
                                   /params['gamma'])

#Mean invasion growth rates normalized by standard deviation
def DelN(args,params):
    R,N,qR,qN = args
    return ((1-params['l'])*params['muc']*params['mug']*R-params['m'])/sigN(args,params)

#Fraction of species that survive in steady state
def phiN(args,params):
    return w0(DelN(args,params))

#Susceptibilities
def nu(args,params):
    R,N,qR,qN = args
    omega_eff = params['omega']+params['muc']*N/params['gamma']
    kappa_eff = params['kappa']+params['l']*params['muc']*N*R/params['gamma']
    r2 = kappa_eff*phiN(args,params)/(params['gamma']*omega_eff*R)
    nu0 = (2*params['gamma']**(-2)*phiN(args,params)**2*kappa_eff/R**2)*(1+np.sqrt(1+(2*r2)**(-2)))
    r1 = kappa_eff*nu0/omega_eff**2
    eps_pk = sigp(args,params)**2/kappa_eff**2
    eps_d = sigd(args,params)**2/omega_eff**2
    return nu0*(1-(6*r1**2*eps_pk-(2*r1-1)*eps_d)/(6*r1**3*r2**(-2)+r1**2*r2**(-2)-16*r1**2-8*r1))

def chi(args,params):
    R,N,qR,qN = args
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
    R,N,qR,qN = args
    return ((1-params['l'])*params['mug']*chi(args,params)*params['sigc']**2*R)**(-1)

#Test satisfaction of competitive exclusion bound for consumers and predators
def test_bound_1(args,params):
    return params['gamma']-phiN(args,params)

def cost_vector(args,params):
    R,N,qR,qN = args
    omega_eff = params['omega']+params['muc']*N/params['gamma']
    kappa_eff = params['kappa']+params['l']*params['muc']*N*R/params['gamma']
    nubar = nu(args,params)*(1-params['l'])*params['mug']*params['sigc']**2/params['gamma']
    r1 = kappa_eff*nubar/omega_eff**2
    eps_pw = sigp(args,params)**2*nubar**2/omega_eff**4
    eps_d = sigd(args,params)**2/omega_eff**2

    RHS = np.asarray([(omega_eff/(2*nubar))*(np.sqrt(1+4*r1)-1-2*(eps_pw-r1*eps_d)/(1+4*r1)**(3/2)),
                     fN(args,params)*sigN(args,params)*w1(DelN(args,params)),
                     (omega_eff**2/(2*nubar**2))*(1+2*r1-np.sqrt(1+4*r1)+eps_d*(1-(1+6*r1)/((1+4*r1)**(3/2)))+2*eps_pw/(1+4*r1)**(3/2)),
                     (fN(args,params)*sigN(args,params))**2*w2(DelN(args,params))])
    
    return RHS

#Return sum of squared differences between RHS and LHS of self-consistency eqns
def cost_function(args,params):
    args = np.exp(args)
    return np.sum((args-cost_vector(args,params))**2)

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
    while fun > eps and k < max_iter:
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
        params = usertools.MakeParams(assumptions)
        params['R0'] = assumptions['R0']
        assumptions['omega'] = 1/assumptions['tau']
        assumptions['sigw'] = 0
        assumptions['sigD'] = np.sqrt((assumptions['sparsity']/(assumptions['sparsity']+1))*((M-1)/M))
        assumptions['kappa'] = np.mean(assumptions['R0']/assumptions['tau'])
        assumptions['mug'] = 1
        assumptions['sigg'] = 0
        N0,R0 = usertools.MakeInitialState(assumptions)

    
        TestPlate = Community([N0,R0],[dNdt,dRdt],params)
        TestPlate.SteadyState()
    
        #Find final states
        Rfinal = TestPlate.R.values.reshape(-1)
        Nfinal = TestPlate.N.values.reshape(-1)
        Nfinal[Nfinal<cutoff] = 0
    
        #Compute moments for feeding in to cavity calculation
        args0 = np.asarray([np.mean(Rfinal), 
                            (Stot*1./S)*np.mean(Nfinal), 
                            np.mean(Rfinal**2), 
                            (Stot*1./S)*np.mean(Nfinal**2)])+1e-10
        
        out = opt.minimize(cost_function,np.log(args0),args=(assumptions,),bounds=(
            (-10,10),(-10,10),(-10,10),(-10,10)))
        args_cav = np.exp(out.x)
        fun = out.fun/np.sum(args0**2)
        k += 1
    if fun > eps:
        args_cav = [np.nan,np.nan,np.nan,np.nan]
        fun = np.nan
    
    if postprocess:
        results_num = {'SphiN':np.sum(Nfinal>0)*1./trials,
                       'M<R>':M*args0[0],
                       'S<N>':S*args0[1],
                       'MqR':M*args0[2],
                       'SqN':S*args0[3]}
        results_cav = {'SphiN':S*phiN(args_cav,assumptions),
                       'M<R>':M*args_cav[0],
                       'S<N>':S*args_cav[1],
                       'MqR':M*args_cav[2],
                       'SqN':S*args_cav[3]}
        return results_num, results_cav, out, args0, assumptions, Rfinal, Nfinal
    else:
        Stot = len(N0)
        new_index = [Stot*['Consumer'],N0.index]
        TestPlate.N.index = new_index
        final_state = TestPlate.N.append(TestPlate.R)
        final_state['Run Number']=run_number
        final_state.set_index('Run Number',append=True,inplace=True)
        final_state = final_state.reorder_levels(['Run Number',0,1])
        final_state.index.names=[None,None,None]
        
        data = pd.DataFrame([args_cav],columns=['<R>','<N>','<R^2>','<N^2>'],index=[run_number])
        data['fun']=fun
        for item in assumptions.keys():
            data[item]=assumptions[item]
        data['S'] = S
        data['M'] = M
        
        sim_index = [Stot*['m']+M*['R0'],list(range(Stot))+list(range(M))]
        sim_params = pd.DataFrame(np.hstack((params['m'],params['R0'])),columns=data.index,index=sim_index).T

        c_matrix = pd.DataFrame(com_params['c'],columns=TestPlate.R.index,index=N0.index)
        c_matrix['Run Number']=run_number
        c_matrix.set_index('Run Number',append=True,inplace=True)
        c_matrix = c_matrix.reorder_levels(['Run Number',0])
        c_matrix.index.names=[None,None]
        
        return data, final_state, sim_params, c_matrix

#############################
#STUDY RESULTS
#############################

param_names = 'K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta,epsN,epsX'.split(',')
idx=pd.IndexSlice
    
def FormatPath(folder):
    if folder==None:
        folder=''
    else:
        if folder != '':
            if folder[-1] != '/':
                folder = folder+'/'
    return folder

def LoadData(folder,task_id = 1,load_all = False, suff = 'K_eta'):
    folder = FormatPath(folder)
    name = '_'+str(task_id)+'_'+suff+'.xlsx'
    finalstate = pd.read_excel(folder+'finalstate'+name,index_col=[0,1,2],header=[0])
    params = pd.read_excel(folder+'data'+name,index_col=[0],header=[0])
    
    if load_all:
        c = pd.read_excel(folder+'cmatrix'+name,index_col=[0,1],header=[0,1])
        simparams = pd.read_excel(folder+'simparams'+name,index_col=[0],header=[0,1])
        return finalstate,params,simparams,c
    else:
        return finalstate,params

def ComputeIPR(df):
    IPR = pd.DataFrame(columns=df.keys(),index=df.index.levels[0])
    for j in df.index.levels[0]:
        p = df.loc[j]/df.loc[j].sum()
        IPR.loc[j] = 1./((p[p>0]**2).sum())
    return IPR

def PostProcess(folders,tmax=10,tmin=1,suff='K',thresh=1e-4):
    j=0
    data_names = ['Herbivore IPR','Plant IPR','Carnivore IPR',
                 'Herbivore richness','Plant richness','Carnivore richness',
                 'Herbivore biomass','Plant biomass','Carnivore biomass']
    data_names = data_names + [name+' Error' for name in data_names]
                        
    for k in np.arange(len(folders)):
        folder = FormatPath(folders[k])

        for task_id in np.arange(tmin,tmax+1):
            data = pd.DataFrame(index=[0],columns=data_names+param_names)
            finalstate,params = LoadData(folder,task_id=task_id, suff=suff)
            N = finalstate.loc[idx[:,'Consumer',:],:]
            R = finalstate.loc[idx[:,'Resource',:],:]
            X = finalstate.loc[idx[:,'Predator',:],:]
            
            N_IPR = ComputeIPR(N)
            R_IPR = ComputeIPR(R)
            X_IPR = ComputeIPR(X)
            n_wells = len(N_IPR.keys())
            
            for plate in N_IPR.index:
                data.loc[j,'Herbivore IPR'] = N_IPR.loc[plate].mean()
                data.loc[j,'Plant IPR'] = R_IPR.loc[plate].mean()
                data.loc[j,'Carnivore IPR'] = X_IPR.loc[plate].mean()
                data.loc[j,'Herbivore richness']=(N.loc[plate]>thresh).sum().mean()
                data.loc[j,'Plant richness']=(R.loc[plate]>thresh).sum().mean()
                data.loc[j,'Carnivore richness']=(X.loc[plate]>thresh).sum().mean()
                data.loc[j,'Herbivore biomass']=N.loc[plate].sum().mean()
                data.loc[j,'Plant biomass']=R.loc[plate].sum().mean()
                data.loc[j,'Carnivore biomass']=X.loc[plate].sum().mean()
                data.loc[j,'Herbivore IPR Error'] = N_IPR.loc[plate].std()/np.sqrt(n_wells)
                data.loc[j,'Plant IPR Error'] = R_IPR.loc[plate].std()/np.sqrt(n_wells)
                data.loc[j,'Carnivore IPR Error'] = X_IPR.loc[plate].std()/np.sqrt(n_wells)
                data.loc[j,'Herbivore richness Error']=(N.loc[plate]>thresh).sum().std()/np.sqrt(n_wells)
                data.loc[j,'Plant richness Error']=(R.loc[plate]>thresh).sum().std()/np.sqrt(n_wells)
                data.loc[j,'Carnivore richness Error']=(X.loc[plate]>thresh).sum().std()/np.sqrt(n_wells)
                data.loc[j,'Herbivore biomass Error']=N.loc[plate].sum().std()/np.sqrt(n_wells)
                data.loc[j,'Plant biomass Error']=R.loc[plate].sum().std()/np.sqrt(n_wells)
                data.loc[j,'Carnivore biomass Error']=X.loc[plate].sum().std()/np.sqrt(n_wells)
                for item in param_names:
                    data.loc[j,item] = params.loc[plate,item]
                j+=1
            data.to_excel('processed_data_'+str(task_id)+'.xlsx')
    return data

def ReviveCommunity(folder,task_id=1,run_number=1,wells=[]):
    finalstate,params,simparams,c=LoadData(folder,task_id=task_id,load_all=True)
    Nold = finalstate.loc[run_number].loc['Consumer']
    Rold = finalstate.loc[run_number].loc[['Resource','Predator']]
    S = len(Nold)
    M = len(Rold)
    if len(wells)==0:
        init_state = [Nold,Rold]
    else:
        init_state = [Nold[wells],Rold[wells]]

    assumptions = {'regulation':'independent','replenishment':'predator','response':'type I'}
    def dNdt(N,R,params):
        return usertools.MakeConsumerDynamics(**assumptions)(N,R,params)
    def dRdt(N,R,params):
        return usertools.MakeResourceDynamics(**assumptions)(N,R,params)
    dynamics = [dNdt,dRdt]

    R0 = simparams.loc[run_number].loc['R0']
    u = simparams.loc[run_number].loc['u']
    params={'c':c.loc[run_number],
            'm':simparams.loc[run_number].loc['m'],
            'u':np.hstack([np.zeros(len(R0)),u]),
            'w':1,
            'g':1,
            'e':1,
            'R0':np.hstack([R0,np.zeros(len(u))]),
            'r':np.hstack([np.ones(len(R0)),np.zeros(len(u))])
           }

    return init_state,dynamics,params
