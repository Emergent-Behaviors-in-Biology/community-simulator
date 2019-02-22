#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:05:55 2017

@author: robertmarsland
"""

import pandas as pd
import numpy as np
from scipy import integrate
import time
import matplotlib.pyplot as plt
try:
    import cvxpy as cvx
except:
    print('cvxpy not installed. Community.SteadyState() not available.')

def IntegrateWell(CommunityInstance,well_info,T0=0,T=1,ns=2,return_all=False,log_time=False,compress_resources=False):
    """
        Integrator for Propagate and TestWell methods of the Community class
        """
    #MAKE LOGARITHMIC TIME AXIS FOR LONG SINGLE RUNS
    if log_time:
        t = 10**(np.linspace(np.log10(T0),np.log10(T0+T),ns))
    else:
        t = np.linspace(T0,T0+T,ns)
    
    #UNPACK INPUT
    y0 = well_info['y0']
    params_comp = well_info['params'].copy()
    
    #COMPRESS STATE AND PARAMETERS TO GET RID OF EXTINCT SPECIES
    S = params_comp['S']
    M = len(y0)-S
    not_extinct = y0>0
    S_comp = np.sum(not_extinct[:S]) #record the new point dividing species from resources
    if not compress_resources:  #only compress resources if we're running non-renewable dynamics
        not_extinct[S:] = True
    not_extinct_idx = np.where(not_extinct)[0]
    y0_comp = y0[not_extinct]
    if 'c' in params_comp.keys():
        params_comp['c']=params_comp['c'][not_extinct[:S],:]
        params_comp['c']=params_comp['c'][:,not_extinct[S:]]
    if 'D' in params_comp.keys():
        params_comp['D']=params_comp['D'][not_extinct[S:],:]
        params_comp['D']=params_comp['D'][:,not_extinct[S:]]
    if 'alpha' in params_comp.keys():
        params_comp['alpha']=params_comp['alpha'][not_extinct[:S],:]
        params_comp['alpha']=params_comp['alpha'][:,not_extinct[:S]]
    for name in ['m','g','K']:
        if name in params_comp.keys():
            if type(params_comp[name]) == np.ndarray:
                assert len(params_comp[name])==S, 'Invalid length for ' + name
                params_comp[name]=params_comp[name][not_extinct[:S]]
    for name in ['e','w','r','tau']:
        if name in params_comp.keys():
            if type(params_comp[name]) == np.ndarray:
                assert len(params_comp[name])==M, 'Invalid length for ' + name
                params_comp[name]=params_comp[name][not_extinct[S:]]

    #INTEGRATE AND RESTORE STATE VECTOR TO ORIGINAL SIZE
    if return_all:
        out = integrate.odeint(CommunityInstance.dydt,y0_comp,t,args=(params_comp,S_comp),mxstep=10000,atol=1e-4)
        traj = np.zeros((np.shape(out)[0],S+M))
        traj[:,not_extinct_idx] = out
        return t, traj
    else:
        out = integrate.odeint(CommunityInstance.dydt,y0_comp,t,args=(params_comp,S_comp),mxstep=10000,atol=1e-4)[-1]
        yf = np.zeros(len(y0))
        yf[not_extinct_idx] = out
        return yf
    
def OptimizeWell(well_info,replenishment='external',tol=1e-7,eps=1,R0t_0=10,verbose=False,max_iters=1000,thresh=np.inf):
    """
    Uses convex optimization to find the steady state of the ecological dynamics.
    """
    
    #UNPACK INPUT
    y0 = well_info['y0'].copy()
    params_comp = well_info['params'].copy()
    N = y0[:params_comp['S']]
    R = y0[params_comp['S']:]
    
    #COMPRESS PARAMETERS TO GET RID OF EXTINCT SPECIES
    not_extinct_consumers = N>0
    if replenishment=='external':
        not_extinct_resources = np.ones(len(R),dtype=bool)
    else:
        not_extinct_resources = R>0
    params_comp['c']=params_comp['c'][not_extinct_consumers,:]
    params_comp['c']=params_comp['c'][:,not_extinct_resources]
    params_comp['D']=params_comp['D'][not_extinct_resources,:]
    params_comp['D']=params_comp['D'][:,not_extinct_resources]
    for name in ['m','g','K']:
        if name in params_comp.keys():
            if type(params_comp[name]) == np.ndarray:
                assert len(params_comp[name])==len(N), 'Invalid length for ' + name
                params_comp[name]=params_comp[name][not_extinct_consumers]
    for name in ['l','w','r','tau']:
        if name in params_comp.keys():
            if type(params_comp[name]) == np.ndarray:
                assert len(params_comp[name])==len(R), 'Invalid length for ' + name
                params_comp[name]=params_comp[name][not_extinct_resources]
    S = len(params_comp['c'])
    M = len(params_comp['c'].T)

    failed = 0
    if params_comp['l'] is not 0:
        assert replenishment == 'external', 'Replenishment must be external for crossfeeding dynamics.'
        
        #Make Q matrix and effective weight vector
        w_mat = np.kron(np.ones((M,1)),np.ones((1,M))*params_comp['w'])
        Q = np.eye(M) - params_comp['l']*params_comp['D']*w_mat/(w_mat.T)
        Qinv = np.linalg.inv(Q)
        Qinv_aa = np.diag(Qinv)
        w = Qinv_aa*(1-params_comp['l'])*params_comp['w']/params_comp['tau']
        Qinv = Qinv - np.diag(Qinv_aa)
        
        #Construct variables for optimizer
        G = params_comp['c']*params_comp['w']*(1-params_comp['l'])/w #Divide by w, because we will multiply by wR
        if isinstance(params_comp['m'],np.ndarray):
            h = params_comp['m'].reshape((S,1))
        else:
            h = np.ones((S,1))*params_comp['m']
        
        #Initialize effective resource concentrations
        R0t = R0t_0*np.ones(M)
        
        #Set up the loop
        Rf = np.inf
        Rf_old = 0
        

        k=0
        ncyc=0
        Delta = 1
        while np.linalg.norm(Rf_old - Rf) > tol and k < max_iters:
            try:
                start_time = time.time()
        
                wR = cvx.Variable(shape=(M,1)) #weighted resources
        
                #Need to multiply by w to get properly weighted KL divergence
                R0t[R0t<0] = 0.01
                wR0 = (R0t*w).reshape((M,1))

                #Solve
                obj = cvx.Minimize(cvx.sum(cvx.kl_div(wR0, wR)))
                constraints = [G*wR <= h, wR >= 0]
                prob = cvx.Problem(obj, constraints)
                prob.solver_stats
                prob.solve(solver=cvx.ECOS,abstol=0.1*tol,reltol=0.1*tol,warm_start=True,verbose=False,max_iters=5000)

                #Record the results
                Rf_old = Rf
                Nf=constraints[0].dual_value[0:S].reshape(S)
                Rf=wR.value.reshape(M)/w

                #Update the effective resource concentrations
                R0t = params_comp['R0'] + Qinv.dot((params_comp['R0']-Rf)/params_comp['tau'])*(params_comp['tau']/Qinv_aa)
                
                Delta_old = Delta
                Delta = np.linalg.norm(Rf_old - Rf)
                if verbose:
                    print('Iteration: '+str(k))
                    print('Delta: '+str(Delta))
                    print('---------------- '+str(time.time()-start_time)[:4]+' s ----------------')
            except:
                #If optimization fails, try new R0t
                shift = eps*np.random.randn(M)
                if np.min(R0t + shift) < 0: #Prevent any values from becoming negative
                    R0t = R0t_0*np.ones(M)
                    Rf = np.inf
                    Rf_old = 0
                else:
                    R0t = R0t + shift
                
                if verbose:
                    print('Added '+str(eps)+' times random numbers')
            k+=1
            #Check for limit cycle
            if np.isfinite(Delta) and Delta > tol and np.abs(Delta-Delta_old) < 0.1*tol:
                ncyc+=1
            if ncyc > 10:
                print('Limit cycle detected')
                k = max_iters

        if k == max_iters:
            failed = 1
                          
    elif params_comp['l'] == 0:
        assert replenishment == 'external', 'Replenishment must be external until quadprog is implemented.'
        
        G = params_comp['c']*params_comp['tau'] #Multiply by tau, because wR has tau in the denominator
        if isinstance(params_comp['m'],np.ndarray):
            h = params_comp['m'].reshape((S,1))
        else:
            h = np.ones((S,1))*params_comp['m']

        wR = cvx.Variable(shape=(M,1)) #weighted resources
        
        #Need to multiply by w to get properly weighted KL divergence
        wR0 = (params_comp['R0']*params_comp['w']/params_comp['tau']).reshape((M,1))

        #Solve
        obj = cvx.Minimize(cvx.sum(cvx.kl_div(wR0, wR)))
        constraints = [G*wR <= h]
        prob = cvx.Problem(obj, constraints)
        prob.solver_stats
        prob.solve(solver=cvx.ECOS,abstol=tol,reltol=tol,warm_start=True,verbose=False,max_iters=200)

        #Record the results
        Nf=constraints[0].dual_value[0:S].reshape(S)
        Rf=wR.value.reshape(M)*params_comp['tau']/params_comp['w']


        
    if not failed:
        N_new = np.zeros(len(N))
        R_new = np.zeros(len(R))
        N_new[np.where(not_extinct_consumers)[0]] = Nf
        R_new[np.where(not_extinct_resources)[0]] = Rf
    else:
        N_new = np.nan*N
        R_new = np.nan*R
        if verbose:
            print('Optimization Failed.')
            
    return np.hstack((N_new, R_new))
    
def TimeStamp(data,t,group='Well'):
    """
    Use Pandas multiindex to record the time that a sample was taken.
    
    data = array to be stamped
    
    t = time
    
    group = orientation of array
    """
    if group == 'Well':
        data_time = data.copy().T
        mdx = pd.MultiIndex.from_product([[t],data_time.index],names=['Time','Well'])
    elif group == 'Species':    
        data_time = data.copy()
        mdx = pd.MultiIndex.from_product([[t],data.index],names=['Time','Species'])
    else:
        return 'Invalid group choice'
    data_time.index = mdx
    return data_time
