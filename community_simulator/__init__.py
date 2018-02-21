#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:02:18 2017

@author: robertmarsland
"""
from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from multiprocessing import Pool
from functools import partial
from .essentialtools import IntegrateWell, TimeStamp

class Community:
    def __init__(self,init_state,dynamics,params,scale=10**9):
        self.N, self.R = init_state

        if not isinstance(self.N, pd.DataFrame):
            column_names = ['W'+str(k) for k in range(np.shape(self.N)[1])]
            species_names = ['S'+str(k) for k in range(np.shape(self.N)[0])]
            self.N = pd.DataFrame(self.N,columns=column_names)
            self.N.index = species_names
            
        if not isinstance(self.R, pd.DataFrame):
            resource_names = ['R'+str(k) for k in range(np.shape(self.R)[0])]
            self.R = pd.DataFrame(self.R,columns=self.N.keys())
            self.R.index = resource_names
            
        self.R0 = self.R.copy()
        self.S, self.n_wells = np.shape(self.N)
        self.M = np.shape(self.R)[0]
        self.dNdt, self.dRdt = dynamics
        
        self.params = params.copy()
        for item in self.params:
            if isinstance(self.params[item],pd.DataFrame):
                self.params[item]=self.params[item].values.squeeze()
            elif isinstance(self.params[item],list):
                self.params[item]=np.asarray(self.params[item])
        if 'D' not in params:
            self.params['D'] = np.ones((self.M,self.M))
            self.params['e'] = 1
        
        self.scale = scale
            
    def dydt(self,y,t,params,S_comp):
        return np.hstack([self.dNdt(y[:S_comp],y[S_comp:],params),
                          self.dRdt(y[:S_comp],y[S_comp:],params)])

    def copy(self):
        return copy.deepcopy(self)
            
    def Reset(self,init_state):
        self.N, self.R = init_state
        
        if not isinstance(self.N, pd.DataFrame):
            column_names = ['D'+str(k) for k in range(np.shape(self.N)[1])]
            species_names = ['S'+str(k) for k in range(np.shape(self.N)[0])]
            self.N = pd.DataFrame(self.N,columns=column_names)
            self.N.index = species_names
            
        if not isinstance(self.R, pd.DataFrame):
            resource_names = ['R'+str(k) for k in range(np.shape(self.R)[0])]
            self.R = pd.DataFrame(self.R,columns=self.N.keys())
            self.R.index = resource_names
            
        self.R0 = self.R.copy()
        self.S, self.n_wells = np.shape(self.N)
        self.M = np.shape(self.R)[0]
            
    def Propagate(self,T,compress_resources=False):
        y_in = self.N.append(self.R).T.values
        IntegrateTheseWells = partial(IntegrateWell,self,self.params,
                                      T=T,compress_resources=compress_resources)
        
        pool = Pool()
        y_out = np.asarray(pool.map(IntegrateTheseWells,y_in)).squeeze().T
        pool.close()
        
        self.N = pd.DataFrame(y_out[:self.S,:],
                              index = self.N.index, columns = self.N.keys())
        self.R = pd.DataFrame(y_out[self.S:,:],
                              index = self.R.index, columns = self.R.keys())
        
    def Passage(self,f_in,scale=None,refresh_resource=True):
        if scale == None:
            scale = self.scale #Use scale from initialization by default
        f = np.asarray(f_in) #Allow for f to be a dataframe
        N_tot = np.sum(self.N)
        R_tot = np.sum(self.R)
        N = np.zeros(np.shape(self.N))
        
        #Perform multinomial sampling on each well to simulate passaging a
        #finite fraction of the discrete collection of cells
        for k in range(self.n_wells):
            for j in range(self.n_wells):
                if f[k,j] > 0 and N_tot[j] > 0:
                    N[:,k] += np.random.multinomial(int(scale*N_tot[j]*f[k,j]),(self.N/N_tot).values[:,j])*1./scale  
        self.N = pd.DataFrame(N, index = self.N.index, columns = self.N.keys())
        
        #In batch culture, there is no need to do multinomial sampling on the resources,
        #since they are externally replenished before they cause numerical problems
        if refresh_resource:
            self.R = pd.DataFrame(np.dot(self.R,f.T), index = self.R.index, columns = self.R.keys())
            self.R = self.R+self.R0
            
        #In continuous culture, it is useful to eliminate the resources that are
        #going extinct, to avoid numerical instability
        else:
            self.R[self.R<0] = 0
            R_tot = np.sum(self.R)
            R = np.zeros(np.shape(self.R))
            for k in range(self.n_wells):
                for j in range(self.n_wells):
                    if f[k,j] > 0 and R_tot[j] > 0:
                        R[:,k] += np.random.multinomial(int(scale*R_tot[j]*f[k,j]),(self.R/R_tot).values[:,j])*1./scale
            self.R = pd.DataFrame(R, index = self.R.index, columns = self.R.keys())
        
    def RunExperiment(self,f,T,np,group='Well',scale=None,refresh_resource=True):
        if scale == None:
            scale = self.scale #Use scale from initialization by default
        t = 0
        N_traj = TimeStamp(self.N,t,group=group)
        R_traj = TimeStamp(self.R,t,group=group)

        for j in range(np):
            self.Passage(f,scale=scale,refresh_resource=refresh_resource)
            self.Propagate(T)
            t += T
            N_traj = N_traj.append(TimeStamp(self.N,t,group=group))
            R_traj = R_traj.append(TimeStamp(self.R,t,group=group))
        
        return N_traj, R_traj
    
    
    def TestWell(self,T = 4,well_name = None,f0 = 1e-3,ns=100,log_time = False,compress_resources=False):
        if well_name == None:
            well_name = self.N.keys()[0]
        N_well = self.N.copy()[well_name] * f0
        R_well = self.R.copy()[well_name]
        t, out = IntegrateWell(self,self.params,N_well.append(R_well).values,T=T,ns=ns,
                               return_all=True,log_time=log_time,compress_resources=compress_resources)
        f, axs = plt.subplots(2,sharex=True)
        Ntraj = out[:,:self.S]
        Rtraj = out[:,self.S:]
        if log_time:
            axs[0].semilogx(t,Ntraj)
            axs[1].semilogx(t,Rtraj)
        else:
            axs[0].plot(t,Ntraj)
            axs[1].plot(t,Rtraj)
        axs[0].set_ylabel('Consumer Abundance')
        axs[1].set_ylabel('Resource Abundance')
        axs[1].set_xlabel('Time')
        plt.show()
        return t, Ntraj, Rtraj
    
    def Metagenome(self):
        c = self.params[0]
        MG = pd.DataFrame(np.dot(c.T,self.N),columns = self.N.keys())
        MG.index = self.R.index
        return MG