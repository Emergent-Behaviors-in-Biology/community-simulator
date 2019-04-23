#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:51:13 2017

@author: robertmarsland
"""

import numpy as np
from scipy.stats import norm
from community_simulator import Community,essentialtools,usertools,visualization
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
def sigX(args,params):
    R,N,X,qR,qN,qX = args
    return np.sqrt(params['sigu']**2 + params['sigd']**2*params['eta']*qN)
def sigN(args,params):
    R,N,X,qR,qN,qX = args
    return np.sqrt(params['sigm']**2 + params['sigc']**2*params['gamma']*qR+params['sigd']**2*qX)
def sigR(args,params):
    R,N,X,qR,qN,qX = args
    return np.sqrt(params['sigK']**2 + params['sigc']**2*qN)

#Mean invasion growth rates normalized by standard deviation
def DelX(args,params):
    R,N,X,qR,qN,qX = args
    return (params['eta']*params['mud']*N-params['u'])/sigX(args,params)
def DelN(args,params):
    R,N,X,qR,qN,qX = args
    return (params['gamma']*params['muc']*R-params['m']-params['mud']*X)/sigN(args,params)
def DelR(args,params):
    R,N,X,qR,qN,qX = args
    return (params['K']-params['muc']*N)/sigR(args,params)

#Fraction of species that survive in steady state
def phiX(args,params):
    return w0(DelX(args,params))
def phiN(args,params):
    return w0(DelN(args,params))
def phiR(args,params):
    return w0(DelR(args,params))

#Factors for converting invasion growth rate to steady-state abundance
def fX(args,params):
    return params['sigc']**2*(params['gamma']*params['eta']*phiR(args,params)+phiX(args,params)
                              -params['eta']*phiN(args,params))/(params['eta']*params['sigd']**2
                                                                 *(params['eta']*phiN(args,params)
                                                                   -phiX(args,params)))
def fN(args,params):
    return ((params['eta']-phiX(args,params)/phiN(args,params))/
            (params['sigc']**2*(params['gamma']*params['eta']*phiR(args,params)+phiX(args,params)
                                -params['eta']*phiN(args,params))))
def fR(args,params):
    return (1+(phiX(args,params)/(params['gamma']*params['eta']*phiR(args,params)))
            -(phiN(args,params)/(params['gamma']*phiR(args,params))))

#Distributions of steady-state abundances
def Xa(args,params,Xvec=np.linspace(0,10,100)):
    return norm.pdf((Xvec/(sigX(args,params)*fX(args,params)))-DelX(args,params))/(sigX(args,params)*fX(args,params))
def Ni(args,params,Nvec=np.linspace(0,10,100)):
    return norm.pdf((Nvec/(sigN(args,params)*fN(args,params)))-DelN(args,params))/(sigN(args,params)*fN(args,params))
def Ral(args,params,Rvec=np.linspace(0,10,100)):
    return norm.pdf((Rvec/(sigR(args,params)*fR(args,params)))-DelR(args,params))/(sigR(args,params)*fR(args,params))

#Susceptibilities
def chiX(args,params):
    return phiX(args,params)*fX(args,params)
def chiN(args,params):
    return phiN(args,params)*fN(args,params)
def chiR(args,params):
    return phiR(args,params)*fR(args,params)

#Test satisfaction of competitive exclusion bound for consumers and predators
def test_bound_1(args,params):
    return params['gamma']*phiR(args,params)-phiN(args,params)+phiX(args,params)/params['eta']
def test_bound_2(args,params):
    return params['eta']*phiN(args,params)-phiX(args,params)

#Return sum of squared differences between RHS and LHS of self-consistency eqns
def cost_function(args,params):
    args_new = args/np.asarray([sigR(args,params)*fR(args,params),
                                        sigN(args,params)*fN(args,params),
                                        sigX(args,params)*fX(args,params),
                                        (sigR(args,params)*fR(args,params))**2,
                                        (sigN(args,params)*fN(args,params))**2,
                                        (sigX(args,params)*fX(args,params))**2])

    RHS = np.asarray([w1(DelR(args,params)),
                     w1(DelN(args,params)),
                     w1(DelX(args,params)),
                     w2(DelR(args,params)),
                     w2(DelN(args,params)),
                     w2(DelX(args,params))])
    
    return np.sum((args_new-RHS)**2)

#Enforce competitive exclusion bounds and keep moments within reasonable values
def cost_function_bounded(args,params):
    args_new = args/np.asarray([sigR(args,params)*fR(args,params),
                                        sigN(args,params)*fN(args,params),
                                        sigX(args,params)*fX(args,params),
                                        (sigR(args,params)*fR(args,params))**2,
                                        (sigN(args,params)*fN(args,params))**2,
                                        (sigX(args,params)*fX(args,params))**2])
    b1 = test_bound_1(args,params)
    b2 = test_bound_2(args,params)
    if np.isfinite(b1) and np.isfinite(b2):
        if b1>0 and b2>0 and np.all(args>=0):
            RHS = np.asarray([w1(DelR(args,params)),
                             w1(DelN(args,params)),
                             w1(DelX(args,params)),
                             w2(DelR(args,params)),
                             w2(DelN(args,params)),
                             w2(DelX(args,params))])
    
            return np.sum((args_new-RHS)**2)
        else:
            return np.inf
    else:
        return np.inf

#############################
#SOLVE PROBLEM
#############################
    
#Generate dynamics for McArthur CRM with predator
assumptions = {'supply':'predator',
               'regulation':'independent',
               'response':'type I'
              }
def dNdt(N,R,params):
    return usertools.MakeConsumerDynamics(assumptions)(N,R,params)
def dRdt(N,R,params):
    return usertools.MakeResourceDynamics(assumptions)(N,R,params)
    
#Generate parameters for McArthur CRM from cavity parameters
def CavityComparison_Gauss(params,S,n_wells=1,Stot=100):
    M = int(round(params['gamma']*S))
    Q = int(round(S/params['eta']))
    
    c_ibeta = params['muc']/S + np.random.randn(Stot,Stot)*params['sigc']/np.sqrt(S)
    d_aj = params['mud']/Q + np.random.randn(Stot,Stot)*params['sigd']/np.sqrt(Q)
    m_i = params['m'] + np.random.randn(Stot)*params['sigm']
    u_a = params['u'] + np.random.randn(Stot)*params['sigu']
    K_alpha = params['K'] + np.random.randn(Stot)*params['sigK']
    
    c_combined = np.hstack((c_ibeta,-d_aj.T))
    r_combined = np.hstack((np.ones(Stot),np.zeros(Stot)))
    K_combined = np.hstack((K_alpha,np.zeros(Stot)))
    w_combined = np.ones(len(r_combined))
    u_combined = np.hstack((np.zeros(Stot),u_a))
    
    N0 = np.zeros((Stot,n_wells))
    R0 = np.zeros((Stot,n_wells))
    X0 = np.zeros((Stot,n_wells))
    
    for k in range(n_wells):
        N0[np.random.choice(Stot,size=S,replace=False),k]=1e-3/S
        R0[np.random.choice(Stot,size=M,replace=False),k]=1
        X0[np.random.choice(Stot,size=Q,replace=False),k]=1e-3/Q 
    
    well_names = ['W'+str(k) for k in range(n_wells)]
    species_names = ['S'+str(k) for k in range(Stot)]
    type_names = ['Resource']*Stot+['Predator']*Stot
    resource_names = 2*list(range(Stot))
    resource_index = [type_names,resource_names]
    N0 = pd.DataFrame(N0,index=species_names,columns=well_names)
    RX0 = pd.DataFrame(np.vstack((R0,X0)),index=resource_index,columns=well_names)

    return [N0,RX0], {'S':S,'Q':Q,'M':M,'c':c_combined,'m':m_i,'R0':K_combined,
           'r':r_combined,'w':w_combined,'u':u_combined,'g':1.}

#Run community to steady state, extract moments of steady state, plot results
def RunCommunity(params,S,plotting=False,com_params={},eps=np.inf,trials=1,postprocess=False,Stot=100,run_number=0):
    assert Stot>=S, 'S must be less than or equal to Stot.'
    assert Stot>=S/params['eta'], 'Q must be less than or equal to Stot.'
    assert Stot>=S*params['gamma'], 'M must be less than or equal to Stot.'
    
    [N0,RX0], com_params_new = CavityComparison_Gauss(params,S,n_wells=trials,Stot=Stot)
    S = com_params_new['S']
    Q = com_params_new['Q']
    M = com_params_new['M']
    
    #Generate new parameter set unless user has passed one
    if len(com_params)==0:
        com_params = com_params_new
    
    #Create Community class instance and run  
    TestPlate = Community([N0,RX0],[dNdt,dRdt],com_params)
    try:
        TestPlate.SteadyState(supply='predator')
    
        #Find final states
        Rfinal = TestPlate.R.loc['Resource'].values.reshape(-1)
        Nfinal = TestPlate.N.values.reshape(-1)
        Xfinal = TestPlate.R.loc['Predator'].values.reshape(-1)
    
        #Compute moments for feeding in to cavity calculation
        args0 = (Stot*1./S)*np.asarray([np.mean(Rfinal)/params['gamma'], 
                                        np.mean(Nfinal), 
                                        np.mean(Xfinal)*params['eta'],
                                        np.mean(Rfinal**2)/params['gamma'], 
                                        np.mean(Nfinal**2), 
                                        np.mean(Xfinal**2)*params['eta']])+1e-10
        if np.mean(Nfinal) == 0:
            args0[0] = w1(params['K']/params['sigK'])*params['sigK']
            args0[3] = w2(params['K']/params['sigK'])*params['sigK']**2
        out = opt.minimize(cost_function_bounded,args0,args=(params,))
        args_cav = out.x
        fun = out.fun
    except:
        args_cav = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
        fun = np.nan
    
    if plotting and fun<=eps:
        f, axs = plt.subplots(3,1,figsize=(12,10))
        
        bins = np.linspace(-200,5,100)
        axs[0].hist(np.log(Rfinal[Rfinal>0]),bins=bins,alpha=0.5,label='Resource')
        axs[0].hist(np.log(Nfinal[Nfinal>0]),bins=bins,alpha=0.5,label='Consumer')
        axs[0].hist(np.log(Xfinal[Xfinal>0]),bins=bins,alpha=0.5,label='Predator')
        axs[0].set_xlabel('Log Species Abundance')

        bins = np.linspace(0,5,20)
        xvec = np.linspace(0,5,100)
        dbin = bins[1]-bins[0]
        axs[1].hist(Rfinal[Rfinal>cutoff],bins=bins,alpha=0.5,color=sns.color_palette()[0],label='Resource')
        axs[1].hist(Nfinal[Nfinal>cutoff],bins=bins,alpha=0.5,color=sns.color_palette()[1],label='Consumer')
        axs[1].hist(Xfinal[Xfinal>cutoff],bins=bins,alpha=0.5,color=sns.color_palette()[2],label='Predator')
        axs[1].plot(xvec,Ral(args_cav,params,Rvec=xvec)*M*dbin*trials,color=sns.color_palette()[0])
        axs[1].plot(xvec,Ni(args_cav,params,Nvec=xvec)*S*dbin*trials,color=sns.color_palette()[1])
        axs[1].plot(xvec,Xa(args_cav,params,Xvec=xvec)*Q*dbin*trials,color=sns.color_palette()[2])
        axs[1].set_xlabel('Non-extinct Species Abundance')
        
        #Compare initial and final values in optimization
        axs[2].plot(xvec,Ral(args0,params,Rvec=xvec)*M,color=sns.color_palette()[0],alpha=0.2)
        axs[2].plot(xvec,Ni(args0,params,Nvec=xvec)*S,color=sns.color_palette()[1],alpha=0.2)
        axs[2].plot(xvec,Xa(args0,params,Xvec=xvec)*Q,color=sns.color_palette()[2],alpha=0.2)
        axs[2].plot(xvec,Ral(args_cav,params,Rvec=xvec)*M,color=sns.color_palette()[0],label='Resource')
        axs[2].plot(xvec,Ni(args_cav,params,Nvec=xvec)*S,color=sns.color_palette()[1],label='Consumer')
        axs[2].plot(xvec,Xa(args_cav,params,Xvec=xvec)*Q,color=sns.color_palette()[2],label='Predator')
        axs[2].set_xlabel('Non-extinct Species Abundance')
        plt.legend()
    
    if postprocess:
        results_num = {'SphiN':np.sum(Nfinal>cutoff)*1./trials,
                       'MphiR':np.sum(Rfinal>cutoff)*1./trials,
                       'QphiX':np.sum(Xfinal>cutoff)*1./trials,
                       'M<R>':M*args0[0],
                       'S<N>':S*args0[1],
                       'Q<X>':Q*args0[2]}
        results_cav = {'SphiN':S*phiN(args_cav,params),
                       'MphiR':M*phiR(args_cav,params),
                       'QphiX':Q*phiX(args_cav,params),
                       'M<R>':M*args_cav[0],
                       'S<N>':S*args_cav[1],
                       'Q<X>':Q*args_cav[2]}
        return results_num, results_cav, out
    else:
        Stot = len(N0)
        new_index = [Stot*['Consumer'],N0.index]
        TestPlate.N.index = new_index
        final_state = TestPlate.N.append(TestPlate.R)
        final_state['Run Number']=run_number
        final_state.set_index('Run Number',append=True,inplace=True)
        final_state = final_state.reorder_levels(['Run Number',0,1])
        final_state.index.names=[None,None,None]
        
        data = pd.DataFrame([args_cav],columns=['<R>','<N>','<X>','<R^2>','<N^2>','<X^2>'],index=[run_number])
        data['fun']=fun
        for item in params.keys():
            data[item]=params[item]
        for item in ['S','M','Q']:
            data[item]=com_params_new[item]
        
        sim_index = [Stot*['m']+Stot*['R0']+Stot*['u'],3*list(range(Stot))]
        sim_params = pd.DataFrame(np.hstack((com_params['m'],com_params['R0'][:Stot],com_params['u'][Stot:])),columns=data.index,index=sim_index).T

        c_matrix = pd.DataFrame(com_params['c'],columns=TestPlate.R.index,index=N0.index)
        c_matrix['Run Number']=run_number
        c_matrix.set_index('Run Number',append=True,inplace=True)
        c_matrix = c_matrix.reorder_levels(['Run Number',0])
        c_matrix.index.names=[None,None]
        
        return data, final_state, sim_params, c_matrix

#############################
#STUDY RESULTS
#############################

param_names = 'K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta'.split(',')
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

def PostProcess(folders,tmax=10,tmin=1):
    j=0
    data_names = ['Consumer IPR','Resource IPR','Predator IPR',
                 'Consumer Richness','Resource Richness','Predator Richness',
                 'Consumer Biomass','Resource Biomass','Predator Biomass']
    
    data = pd.DataFrame(index=[0],columns=data_names+param_names)
                        
    for k in np.arange(len(folders)):
        folder = FormatPath(folders[k])

        for task_id in np.arange(tmin,tmax+1):
            finalstate,params = LoadData(folder,task_id=task_id)
            N = finalstate.loc[idx[:,'Consumer',:],:]
            R = finalstate.loc[idx[:,'Resource',:],:]
            X = finalstate.loc[idx[:,'Predator',:],:]
            
            N_IPR = ComputeIPR(N)
            R_IPR = ComputeIPR(R)
            X_IPR = ComputeIPR(X)
    
            for plate in N_IPR.index:
                for well in N_IPR.keys():
                    data.loc[j,'Consumer IPR'] = N_IPR.loc[plate,well]
                    data.loc[j,'Resource IPR'] = R_IPR.loc[plate,well]
                    data.loc[j,'Predator IPR'] = X_IPR.loc[plate,well]
                    data.loc[j,'Consumer Richness']=(N.loc[plate]>0).sum()[well]
                    data.loc[j,'Resource Richness']=(R.loc[plate]>0).sum()[well]
                    data.loc[j,'Predator Richness']=(X.loc[plate]>0).sum()[well]
                    data.loc[j,'Consumer Biomass']=N[well].loc[plate].sum()
                    data.loc[j,'Resource Biomass']=R[well].loc[plate].sum()
                    data.loc[j,'Predator Biomass']=X[well].loc[plate].sum()
                    for item in param_names:
                        data.loc[j,item] = params.loc[plate,item]
                    
                    j+=1
            data.to_excel('processed_data.xlsx')
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
