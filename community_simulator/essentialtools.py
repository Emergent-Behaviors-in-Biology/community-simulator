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
    cvxpy_installed = True
except:
    print('cvxpy not installed. Community.SteadyState() not available.')
    cvxpy_installed = False
    
def CompressParams(not_extinct_consumers,not_extinct_resources,params,dimensions,S,M):
    params_comp = params.copy()
    if 'SxM' in dimensions.keys():
        for item in dimensions['SxM']:
            if item in params_comp.keys():
                assert np.shape(params_comp[item])==(S,M), 'Invalid shape for ' + item + '. Please update dimensions dictionary with correct dimensions.'
                params_comp[item]=params_comp[item][not_extinct_consumers,:]
                params_comp[item]=params_comp[item][:,not_extinct_resources]
    if 'MxM' in dimensions.keys():
        for item in dimensions['MxM']:
            if item in params_comp.keys():
                assert np.shape(params_comp[item])==(M,M), 'Invalid shape for ' + item + '. Please update dimensions dictionary with correct dimensions.'
                params_comp[item]=params_comp[item][not_extinct_resources,:]
                params_comp[item]=params_comp[item][:,not_extinct_resources]
    if 'SxS' in dimensions.keys():
        for item in dimensions['SxS']:
            if item in params_comp.keys():
                assert np.shape(params_comp[item])==(S,S), 'Invalid shape for ' + item + '. Please update dimensions dictionary with correct dimensions.'
                params_comp[item]=params_comp[item][not_extinct_consumers,:]
                params_comp[item]=params_comp[item][:, not_extinct_consumers]
    if 'S' in dimensions.keys():
        for item in dimensions['S']:
            if item in params_comp.keys():
                if type(params_comp[item]) == np.ndarray:
                    assert len(params_comp[item])==S, 'Invalid length for ' + item + '. Please update dimensions dictionary with correct dimensions.'
                    params_comp[item]=params_comp[item][not_extinct_consumers]
    if 'M' in dimensions.keys():
        for item in dimensions['M']:
            if item in params_comp.keys():
                if type(params_comp[item]) == np.ndarray:
                    assert len(params_comp[item])==M, 'Invalid length for ' + item + '. Please update dimensions dictionary with correct dimensions.'
                    params_comp[item]=params_comp[item][not_extinct_resources]  
    return params_comp

def IntegrateWell(CommunityInstance,well_info,T0=0,T=1,ns=2,return_all=False,log_time=False,
                  compress_resources=False,compress_species=True):
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
    
    #COMPRESS STATE AND PARAMETERS TO GET RID OF EXTINCT SPECIES
    S = well_info['params']['S']
    M = len(y0)-S
    not_extinct = y0>0
    if not compress_species:
        not_extinct[:S] = True
    if not compress_resources:  #only compress resources if we're running non-renewable dynamics
        not_extinct[S:] = True
    S_comp = np.sum(not_extinct[:S]) #record the new point dividing species from resources
    not_extinct_idx = np.where(not_extinct)[0]
    y0_comp = y0[not_extinct]
    not_extinct_consumers = not_extinct[:S]
    not_extinct_resources = not_extinct[S:]

    #Compress parameters
    params_comp = CompressParams(not_extinct_consumers,not_extinct_resources,well_info['params'],CommunityInstance.dimensions,S,M)

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
    
def OptimizeWell(well_info,supply='external',tol=1e-7,shift_size=1,eps=1e-20,
                 alpha=0.5,R0t_0=10,verbose=False,max_iters=1000,dimensions={}):
    """
    Uses convex optimization to find the steady state of the ecological dynamics.
    """

    assert cvxpy_installed, 'CVXPY not found. To use SteadyState(), please download and install CVXPY from www.cvxpy.org.'
    
    #UNPACK INPUT
    y0 = well_info['y0'].copy()
    S = well_info['params']['S']
    M = len(y0)-S
    N = y0[:S]
    R = y0[S:]
    
    #COMPRESS PARAMETERS TO GET RID OF EXTINCT SPECIES
    assert np.isnan(N).sum() == 0, 'All species must have numeric abundances (not NaN).'
    assert np.isnan(R).sum() == 0, 'All resources must have numeric abundances (not NaN).'
    not_extinct_consumers = N>0
    if supply=='external':
        not_extinct_resources = np.ones(len(R),dtype=bool)
    else:
        not_extinct_resources = R>0
    #Compress parameters
    params_comp = CompressParams(not_extinct_consumers,not_extinct_resources,well_info['params'],dimensions,S,M)    
    S = len(params_comp['c'])
    M = len(params_comp['c'].T)

    failed = 0
    if np.max(params_comp['l']) != 0:
        assert supply == 'external', 'Replenishment must be external for crossfeeding dynamics.'
        
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
        if len(np.shape(R0t_0)) == 0:
            R0t = R0t_0*np.ones(M)
        else:
            R0t = R0t_0
        
        #Set up the loop
        Rf = np.inf
        Rf_old = 0
        

        k=0
        ncyc=0
        Delta = 1
        Delta_old = 1
        while np.linalg.norm(Rf_old - Rf) > tol and k < max_iters:
            try:
                start_time = time.time()
        
                wR = cvx.Variable(shape=(M,1)) #weighted resources
        
                #Need to multiply by w to get properly weighted KL divergence
                R0t = np.sqrt(R0t**2+eps)
                wR0 = (R0t*w).reshape((M,1))

                #Solve
                obj = cvx.Minimize(cvx.sum(cvx.kl_div(wR0, wR)))
                constraints = [G@wR <= h, wR >= 0]
                prob = cvx.Problem(obj, constraints)
                prob.solver_stats
                prob.solve(solver=cvx.ECOS,abstol=1e-8,reltol=1e-8,warm_start=True,verbose=False,max_iters=5000)

                #Record the results
                Rf_old = Rf
                Nf=constraints[0].dual_value[0:S].reshape(S)
                Rf=wR.value.reshape(M)/w

                #Update the effective resource concentrations
                R0t_new = params_comp['R0'] + Qinv.dot((params_comp['R0']-Rf)/params_comp['tau'])*(params_comp['tau']/Qinv_aa)
                Delta_R0t = R0t_new-R0t
                R0t = R0t + alpha*Delta_R0t
                
                Delta_old = Delta
                Delta = np.linalg.norm(Rf_old - Rf)
                if verbose:
                    print('Iteration: '+str(k))
                    print('Delta: '+str(Delta))
                    print('---------------- '+str(time.time()-start_time)[:4]+' s ----------------')
            except:
                #If optimization fails, try new R0t
                shift = shift_size*np.random.randn(M)
                R0t = np.abs(R0t + shift)
                
                if verbose:
                    print('Added '+str(shift_size)+' times random numbers')
            k+=1
            #Check for limit cycle
            if np.isfinite(Delta) and Delta > tol and np.abs(Delta-Delta_old) < 0.1*tol:
                ncyc+=1
            if ncyc > 10:
                print('Limit cycle detected')
                k = max_iters

        if k == max_iters:
            print('Maximum iterations exceeded. Try decreasing alpha to improve convergence.')
            failed = 1
        else:
            if verbose:
                print('success')
                          
    elif params_comp['l'] == 0:
        if supply == 'external':
            G = params_comp['c']*params_comp['tau'] #Multiply by tau, because wR has tau in the denominator
            if isinstance(params_comp['m'],np.ndarray):
                h = params_comp['m'].reshape((S,1))
            else:
                h = np.ones((S,1))*params_comp['m']

            wR = cvx.Variable(shape=(M,1)) #weighted resources
        
            #Need to multiply by w to get properly weighted KL divergence
            wR0 = (params_comp['R0']*params_comp['w']*np.ones(M)/params_comp['tau']).reshape((M,1))

            #Solve
            obj = cvx.Minimize(cvx.sum(cvx.kl_div(wR0, wR)))
            constraints = [G@wR <= h]
            prob = cvx.Problem(obj, constraints)
            prob.solver_stats
            prob.solve(solver=cvx.ECOS,abstol=1e-8,reltol=1e-8,warm_start=True,verbose=False,max_iters=5000)

            #Record the results
            Nf=constraints[0].dual_value[0:S].reshape(S)
            Rf=wR.value.reshape(M)*params_comp['tau']/params_comp['w']
    
        elif supply == 'self-renewing':
            #Format constants and variables
            if isinstance(params_comp['m'],np.ndarray):
                h = params_comp['m']
            else:
                h = np.ones(S)*params_comp['m']
            if isinstance(params_comp['w'],np.ndarray):
                w = params_comp['w']
            else:
                w = np.ones(M)*params_comp['w']
            if isinstance(params_comp['r'],np.ndarray):
                r = params_comp['r']
            else:
                r = np.ones(M)*params_comp['r']
            R0 = params_comp['R0']
            R_opt = cvx.Variable(M)
            
            #Make constraints
            G = params_comp['c']*params_comp['w']
            G = np.vstack((G,-np.eye(M)))
            h = np.hstack((h,np.zeros(M)))

            #Solve
            obj = cvx.Minimize((1/2)*cvx.quad_form(R0-R_opt,np.diag(w*r)))
            constraints = [G@R_opt <= h]
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.ECOS,abstol=1e-9,feastol=1e-7,abstol_inacc=1e-5,
                       feastol_inacc=1e-5,max_iters=5000)
        
            #Record the results
            Nf=constraints[0].dual_value[:S]
            Rf=R_opt.value
        elif supply == 'predator':
            #Format constants and variables
            if isinstance(params_comp['m'],np.ndarray):
                h = params_comp['m']
            else:
                h = np.ones(S)*params_comp['m']
            if isinstance(params_comp['w'],np.ndarray):
                w = params_comp['w']
            else:
                w = np.ones(M)*params_comp['w']
            r = params_comp['r']
            u = params_comp['u']
            R0 = params_comp['R0']
            R_opt = cvx.Variable(M)
            
            #Make constraints
            G = params_comp['c']*params_comp['w']
            G = np.vstack((G,-np.eye(M)))
            h = np.hstack((h,np.zeros(M)))
            
            #Solve
            obj = cvx.Minimize((1/2)*cvx.quad_form(R0-R_opt,np.diag(w*r))+u.T@R_opt)
            constraints = [G@R_opt <= h]
            prob = cvx.Problem(obj, constraints)
            prob.solve(solver=cvx.ECOS,abstol=1e-9,feastol=1e-7,abstol_inacc=1e-5,
                       feastol_inacc=1e-5,max_iters=5000)
            
            #Record the results
            Nf=constraints[0].dual_value[:S]
            Rf=R_opt.value
        else:
            print('supply must be external or self-renewing')
            failed = True
        
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
