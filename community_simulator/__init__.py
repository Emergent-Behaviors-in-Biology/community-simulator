#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:02:18 2017

@author: robertmarsland
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from multiprocessing import Pool
from functools import partial
from essentialtools import IntegrateDeme, TimeStamp

class Community:
    def __init__(self,init_state,dynamics,params):
        self.N, self.R = init_state
        self.R0 = self.R.copy()
        self.params = params
        self.dNdt, self.dRdt = dynamics
        self.S, self.A = np.shape(self.N)
        self.M = np.shape(self.R)[0]
    
        if not isinstance(self.N, pd.DataFrame):
            column_names = ['D'+str(k) for k in range(np.shape(self.N)[1])]
            species_names = ['S'+str(k) for k in range(np.shape(self.N)[0])]
            self.N = pd.DataFrame(self.N,columns=column_names)
            self.N.index = species_names
            
        if not isinstance(self.R0, pd.DataFrame):
            resource_names = ['R'+str(k) for k in range(np.shape(self.R)[0])]
            self.R = pd.DataFrame(self.R,columns=self.N.keys())
            self.R.index = resource_names
            
    def dydt(self,y,t):
        return np.hstack([self.dNdt(y[:self.S],y[self.S:],self.params),
                          self.dRdt(y[:self.S],y[self.S:],self.params)])

    def copy(self):
        return copy.deepcopy(self)
            
    def Reset(self,init_state):
        self.N, self.R = init_state
        self.R0 = self.R.copy()
        self.S, self.A = np.shape(self.N)
        self.M = np.shape(self.R)[0]
    
        if not isinstance(self.N, pd.DataFrame):
            column_names = ['D'+str(k) for k in range(np.shape(self.N)[1])]
            species_names = ['S'+str(k) for k in range(np.shape(self.N)[0])]
            self.N = pd.DataFrame(self.N,columns=column_names)
            self.N.index = species_names
            
        if not isinstance(self.R0, pd.DataFrame):
            resource_names = ['R'+str(k) for k in range(np.shape(self.R)[0])]
            self.R = pd.DataFrame(self.R,columns=self.N.keys())
            self.R.index = resource_names
            
    def Propagate(self,T):
        y_in = self.N.append(self.R).T.values
        IntegrateTheseDemes = partial(IntegrateDeme,self,T=T)
        
        pool = Pool()
        y_out = np.asarray(pool.map(IntegrateTheseDemes,y_in)).squeeze().T
        pool.close()
        
        self.N = pd.DataFrame(y_out[:self.S,:],
                              index = self.N.index, columns = self.N.keys())
        self.R = pd.DataFrame(y_out[self.S:,:],
                              index = self.R.index, columns = self.R.keys())
        
    def Dilute(self,f_in,scale=10**9):
        f = np.asarray(f_in) #Allow for f to be a dataframe
        N_tot = np.sum(self.N)
        N = np.zeros(np.shape(self.N))
        for k in range(self.A):
            for j in range(self.A):
                if f[k,j] > 0:
                    N[:,k] += np.random.multinomial(int(scale*N_tot[j]*f[k,j]),(self.N/N_tot).values[:,j])*1./scale
            
        self.N = pd.DataFrame(N, index = self.N.index, columns = self.N.keys())
        self.R = self.R0 + pd.DataFrame(np.dot(self.R,f), index = self.R.index, columns = self.R.keys())
        
    def RunExperiment(self,f,T,np,group='Deme',scale=10**9):
        t = 0
        N_traj = TimeStamp(self.N,t,group=group)
        R_traj = TimeStamp(self.R,t,group=group)

        for j in range(np):
            self.Dilute(f,scale=scale)
            self.Propagate(T)
            t += T
            N_traj = N_traj.append(TimeStamp(self.N,t,group=group))
            R_traj = R_traj.append(TimeStamp(self.R,t,group=group))
        
        return N_traj, R_traj
    
    
    def TestDeme(self,T = 4,DemeName = None,f0 = 1e-3):
        if DemeName == None:
            DemeName = self.N.keys()[0]
        N0 = self.N.copy()[DemeName] * f0
        R0 = self.R.copy()[DemeName]
        t, out = IntegrateDeme(self,N0.append(R0),T=T,ns=100,return_all=True)
        f, axs = plt.subplots(2,sharex=True)
        Ntraj = out[:,:self.S]
        Rtraj = out[:,self.S:]
        axs[0].plot(t,Ntraj)
        axs[0].set_ylabel('Species Abundance')
        axs[1].plot(t,Rtraj)
        axs[1].set_ylabel('Resource Abundance')
        axs[1].set_xlabel('Time')
        plt.show()
        return t, Ntraj, Rtraj
    
    def Metagenome(self):
        c = self.params[0]
        MG = pd.DataFrame(np.dot(c.T,self.N),columns = self.N.keys())
        MG.index = self.R.index
        return MG