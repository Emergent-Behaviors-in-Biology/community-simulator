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
def test_bound_1(log_args,params):
    args = np.exp(log_args)
    return params['gamma']*phiR(args,params)-phiN(args,params)+phiX(args,params)/params['eta']
def test_bound_2(log_args,params):
    args = np.exp(log_args)
    return params['eta']*phiN(args,params)-phiX(args,params)

#Return sum of squared differences between RHS and LHS of self-consistency eqns
def cost_function(log_args,params):
    args = np.exp(log_args)
    RHS_R = sigR(args,params)*fR(args,params)*w1(DelR(args,params))
    RHS_N = sigN(args,params)*fN(args,params)*w1(DelN(args,params))
    RHS_X = sigX(args,params)*fX(args,params)*w1(DelX(args,params))
    RHS_qR = (sigR(args,params)*fR(args,params))**2 * w2(DelR(args,params))
    RHS_qN = (sigN(args,params)*fN(args,params))**2 * w2(DelN(args,params))
    RHS_qX = (sigX(args,params)*fX(args,params))**2 * w2(DelX(args,params))

    RHS = np.asarray([RHS_R,RHS_N,RHS_X,RHS_qR,RHS_qN,RHS_qX])
    
    return np.sum((args-RHS)**2)

#Enforce competitive exclusion bounds and keep moments within reasonable values
def cost_function_bounded(log_args,params,upper_bound):
    args = np.exp(log_args)
    b1 = test_bound_1(log_args,params)
    b2 = test_bound_2(log_args,params)
    log_arg_max = np.max(log_args)
    if np.isfinite(b1) and np.isfinite(b2) and (log_arg_max < upper_bound):
        if b1>0 and b2>0:
            RHS_R = sigR(args,params)*fR(args,params)*w1(DelR(args,params))
            RHS_N = sigN(args,params)*fN(args,params)*w1(DelN(args,params))
            RHS_X = sigX(args,params)*fX(args,params)*w1(DelX(args,params))
            RHS_qR = (sigR(args,params)*fR(args,params))**2 * w2(DelR(args,params))
            RHS_qN = (sigN(args,params)*fN(args,params))**2 * w2(DelN(args,params))
            RHS_qX = (sigX(args,params)*fX(args,params))**2 * w2(DelX(args,params))

            RHS = np.asarray([RHS_R,RHS_N,RHS_X,RHS_qR,RHS_qN,RHS_qX])
    
            return np.sum((args-RHS)**2)
        else:
            return np.inf
    else:
        return np.inf
    
#Generate dynamics for McArthur CRM with predator
assumptions = {'replenishment':'predator'}
def dNdt(N,R,params):
    return usertools.MakeConsumerDynamics(**assumptions)(N,R,params)
def dRdt(N,R,params):
    return usertools.MakeResourceDynamics(**assumptions)(N,R,params)
    
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
def RunCommunity(params,S,T=10,n_iter=800,plotting=False,com_params={},log_bound=15,
                 cutoff=1e-10,eps=np.inf,trials=1,postprocess=False,Stot=100,run_number=0):
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
    Ntraj,RXtraj = TestPlate.RunExperiment(np.eye(np.shape(N0)[1]),T,n_iter,
                                           refresh_resource=False,scale=1./cutoff)
    Rtraj = RXtraj['Resource']
    Xtraj = RXtraj['Predator']
    
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
    out = opt.minimize(cost_function_bounded,np.log(args0),args=(params,log_bound))
    args_cav = np.exp(out.x)
    
    if plotting and out.fun<=eps:
        plotting_well = N0.keys()[0]
        f, axs = plt.subplots(3,2,figsize=(12,10))
        visualization.StackPlot(Ntraj.loc(axis=0)[:,plotting_well].T,ax=axs[0,0])
        visualization.StackPlot(Rtraj.loc(axis=0)[:,plotting_well].T,ax=axs[1,0])
        visualization.StackPlot(Xtraj.loc(axis=0)[:,plotting_well].T,ax=axs[2,0])
        axs[2,0].set_xlabel('Time')
        axs[0,0].set_ylabel('Consumer Abundance')
        axs[1,0].set_ylabel('Resource Abundance')
        axs[2,0].set_ylabel('Predator Abundance')
        
        bins = np.linspace(-200,5,100)
        axs[0,1].hist(np.log(Rfinal[Rfinal>0]),bins=bins,alpha=0.5,label='Resource')
        axs[0,1].hist(np.log(Nfinal[Nfinal>0]),bins=bins,alpha=0.5,label='Consumer')
        axs[0,1].hist(np.log(Xfinal[Xfinal>0]),bins=bins,alpha=0.5,label='Predator')
        axs[0,1].set_xlabel('Log Species Abundance')

        bins = np.linspace(0,5,20)
        xvec = np.linspace(0,5,100)
        dbin = bins[1]-bins[0]
        axs[1,1].hist(Rfinal[Rfinal>cutoff],bins=bins,alpha=0.5,color=sns.color_palette()[0],label='Resource')
        axs[1,1].hist(Nfinal[Nfinal>cutoff],bins=bins,alpha=0.5,color=sns.color_palette()[1],label='Consumer')
        axs[1,1].hist(Xfinal[Xfinal>cutoff],bins=bins,alpha=0.5,color=sns.color_palette()[2],label='Predator')
        axs[1,1].plot(xvec,Ral(args_cav,params,Rvec=xvec)*M*dbin*trials,color=sns.color_palette()[0])
        axs[1,1].plot(xvec,Ni(args_cav,params,Nvec=xvec)*S*dbin*trials,color=sns.color_palette()[1])
        axs[1,1].plot(xvec,Xa(args_cav,params,Xvec=xvec)*Q*dbin*trials,color=sns.color_palette()[2])
        axs[1,1].set_xlabel('Non-extinct Species Abundance')
        
        #Compare initial and final values in optimization
        axs[2,1].plot(xvec,Ral(args0,params,Rvec=xvec)*M,color=sns.color_palette()[0],alpha=0.2)
        axs[2,1].plot(xvec,Ni(args0,params,Nvec=xvec)*S,color=sns.color_palette()[1],alpha=0.2)
        axs[2,1].plot(xvec,Xa(args0,params,Xvec=xvec)*Q,color=sns.color_palette()[2],alpha=0.2)
        axs[2,1].plot(xvec,Ral(args_cav,params,Rvec=xvec)*M,color=sns.color_palette()[0],label='Resource')
        axs[2,1].plot(xvec,Ni(args_cav,params,Nvec=xvec)*S,color=sns.color_palette()[1],label='Consumer')
        axs[2,1].plot(xvec,Xa(args_cav,params,Xvec=xvec)*Q,color=sns.color_palette()[2],label='Predator')
        axs[2,1].set_xlabel('Non-extinct Species Abundance')
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
        data['fun']=out.fun
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
