#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:51:13 2017

@author: robertmarsland
"""

import numpy as np
from scipy.stats import norm
from community_simulator import Community, models, essentialtools
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt

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
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return np.sqrt(sigu**2 + sigd**2*eta*qN)
def sigN(args,params):
    R,N,X,qR,qN,qX = args
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return np.sqrt(sigm**2 + sigc**2*gamma*qR+sigd**2*qX)
def sigR(args,params):
    R,N,X,qR,qN,qX = args
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return np.sqrt(sigK**2 + sigc**2*qN)

#Mean invasion growth rates normalized by standard deviation
def DelX(args,params):
    R,N,X,qR,qN,qX = args
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return (eta*mud*N-u)/sigX(args,params)
def DelN(args,params):
    R,N,X,qR,qN,qX = args
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return (gamma*muc*R-m-mud*X)/sigN(args,params)
def DelR(args,params):
    R,N,X,qR,qN,qX = args
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return (K-muc*N)/sigR(args,params)

#Fraction of species that survive in steady state
def phiX(args,params):
    return w0(DelX(args,params))
def phiN(args,params):
    return w0(DelN(args,params))
def phiR(args,params):
    return w0(DelR(args,params))

#Factors for converting invasion growth rate to steady-state abundance
def fX(args,params):
    R,N,X,qR,qN,qX = args
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return sigc**2*(gamma*eta*phiR(args,params)+phiX(args,params)
                    -eta*phiN(args,params))/(eta*sigd**2*(eta*phiN(args,params)-phiX(args,params)))
def fN(args,params):
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return (eta-phiX(args,params)/phiN(args,params))/(sigc**2*(gamma*eta*phiR(args,params)+phiX(args,params)
                                                               -eta*phiN(args,params)))
def fR(args,params):
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return 1+(phiX(args,params)/(gamma*eta*phiR(args,params)))-(phiN(args,params)/(gamma*phiR(args,params)))

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
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return gamma*phiR(args,params)-phiN(args,params)+phiX(args,params)/eta
def test_bound_2(log_args,params):
    args = np.exp(log_args)
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = params
    return eta*phiN(args,params)-phiX(args,params)

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
def cost_function_bounded(log_args,params,bounds):
    args = np.exp(log_args)
    b1 = test_bound_1(log_args,params)
    b2 = test_bound_2(log_args,params)
    log_arg_max = np.max(log_args)
    log_arg_min = np.min(log_args)
    if np.isfinite(b1) and np.isfinite(b2) and (log_arg_max < bounds[1]) and (log_arg_min > bounds[0]):
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
    
#Generate parameters for McArthur CRM from cavity parameters
def CavityComparison_Gauss(params,S,n_demes=1):
    K,sigK,muc,sigc,mud,sigd,m,sigm,u,sigu,gamma,eta = np.asarray(params,dtype=float)
    M = int(round(gamma*S))
    Q = int(round(S/eta))
    
    c_ibeta = muc/S + np.random.randn(S,M)*sigc/np.sqrt(S)
    d_aj = mud/Q + np.random.randn(Q,S)*sigd/np.sqrt(Q)
    m_i = m + np.random.randn(S)*sigm
    u_a = u + np.random.randn(Q)*sigu
    K_alpha = K + np.random.randn(M)*sigK
    
    c_combined = np.hstack((c_ibeta,-d_aj.T))
    r_combined = np.hstack((K_alpha,-u_a))
    Kinv_combined = np.hstack((1./K_alpha,np.zeros(Q)))
    w_combined = np.ones(len(r_combined))
    
    N0 = np.ones((S,n_demes))*1e-3/S
    R0 = np.ones((M,n_demes))
    X0 = np.ones((Q,n_demes))*1e-3/Q

    return [N0,R0,X0], {'c':c_combined,'m':m_i,'Kinv':Kinv_combined,
           'r':r_combined,'w':w_combined}

#Run community to steady state, extract moments of steady state, plot results
def RunCommunity(params,S,init_state=[],T=100,ns=8000,log_time=True,plotting=True,
                 com_params={},log_bounds=[-15,15]):
    logmin,logmax = log_bounds
    [N0,R0,X0], com_params_new = CavityComparison_Gauss(params,S)
    M = np.shape(R0)[0]
    Q = np.shape(X0)[0]
    
    #Generate new parameter set unless user has passed one
    if len(com_params)==0:
        com_params = com_params_new
    Batch = Community([N0,np.vstack((R0,X0))],[models.dNdt_CRM,models.dRdt_CRM],com_params)
    
    #Use standard initial condition unless user has passed one
    if len(init_state)==0:
        init_state = np.hstack((N0[:,0],R0[:,0],X0[:,0]))
    else:
        #If user has passed an initial condition, remove very low abundance
        #species, and attempt an invasion by all species. This ensures stability
        #and convergence to true steady state
        init_state[init_state<1e-20] = 0
        init_state = init_state + np.hstack((N0[:,0],1e-3*R0[:,0],X0[:,0]))
        
    #Integrate
    t, out = essentialtools.IntegrateWell(Batch,init_state,T=T,ns=ns,return_all=True,log_time=log_time)
    
    #Extract results into separate vectors for the three levels
    Ntraj = out[:,:S]
    Rtraj = out[:,S:S+Q]
    Xtraj = out[:,S+Q:]
    
    #Find final states
    Rfinal = Rtraj[-1,:]
    Nfinal = Ntraj[-1,:]
    Xfinal = Xtraj[-1,:]
    
    #Compute moments for feeding in to cavity calculation
    args0 = [np.mean(Rfinal), np.mean(Nfinal), np.mean(Xfinal),
            np.mean(Rfinal**2), np.mean(Nfinal**2), np.mean(Xfinal**2)]
    
    out = opt.minimize(cost_function_bounded,np.log(args0),args=(params,[logmin,logmax]))
    args_cav = np.exp(out.x)
    
    if plotting:
        f, axs = plt.subplots(3,2,figsize=(12,10))
        axs[0,0].plot(t,Ntraj)
        axs[1,0].plot(t,Rtraj)
        axs[2,0].plot(t,Xtraj)
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
        axs[1,1].hist(Rfinal[Rfinal>1e-10],bins=bins,alpha=0.5,color=sns.color_palette()[0],label='Resource')
        axs[1,1].hist(Nfinal[Nfinal>1e-10],bins=bins,alpha=0.5,color=sns.color_palette()[1],label='Consumer')
        axs[1,1].hist(Xfinal[Xfinal>1e-10],bins=bins,alpha=0.5,color=sns.color_palette()[2],label='Predator')
        axs[1,1].plot(xvec,Ral(args_cav,params,Rvec=xvec)*M*dbin,color=sns.color_palette()[0])
        axs[1,1].plot(xvec,Ni(args_cav,params,Nvec=xvec)*S*dbin,color=sns.color_palette()[1])
        axs[1,1].plot(xvec,Xa(args_cav,params,Xvec=xvec)*Q*dbin,color=sns.color_palette()[2])
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
    
        
    return [Nfinal,Rfinal,Xfinal], com_params, args0, out