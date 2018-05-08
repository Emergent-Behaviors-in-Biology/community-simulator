#!/usr/bin/env python3
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
        """
        Initialize a new "96-well plate" for growing microbial communities.
        
        init_state = [N0,R0] where N0 and R0 are 2-D arrays specifying the 
            initial consumer and resource concentrations, respectively, in each
            of the wells. Each species of consumer and each resource has its
            own row, and each well has its own column. If N0 and R0 are Pandas
            DataFrames, the row and column labels will be preserved throughout
            all subsequent calculations. Otherwise, standard row and column
            labels will be automatically supplied.
        
        dynamics = [dNdt,dRdt] where dNdt(N,R,params) and dRdt(N,R,params) are 
            vectorized functions of the consumer and resource concentrations
            N and R for a single well. params is a Python dictionary containing
            the parameters that required by these functions, and is passed to 
            the new plate instance in the next argument. 
            
        params was just explained above. Note that the integrator IntegrateWell
            defined in essentialtools.py assumes that the model has no species-
            specific parameters other than those employed in the supplied 
            function constructor found in usertools.py. If additional or different
            parameters are required, IntegrateWell must be appropriately modified.
            
        scale is a conversion factor specifying the number of individual microbial 
            cells present when N = 1. It is used in the Passage method defined 
            below to perform multinomial sampling, and controls the strength
            of population noise. 
        """
        #SAVE INITIAL STATE
        N, R = init_state
        if not isinstance(N, pd.DataFrame):#add labels to consumer state
            if len(np.shape(N)) == 1:
                N = N[:,np.newaxis]
            column_names = ['W'+str(k) for k in range(np.shape(N)[1])]
            species_names = ['S'+str(k) for k in range(np.shape(N)[0])]
            N = pd.DataFrame(N,columns=column_names)
            N.index = species_names
        if not isinstance(R, pd.DataFrame):#add labels to resource state
            if len(np.shape(R)) == 1:
                R = R[:,np.newaxis]
            resource_names = ['R'+str(k) for k in range(np.shape(R)[0])]
            R = pd.DataFrame(R,columns=N.keys())
            R.index = resource_names
        self.N = N.copy()
        self.R = R.copy()
        self.R0 = R.copy() #(for refreshing media on passaging if "refresh_resource" is turned on)
        self.S, self.n_wells = np.shape(self.N)
        self.M = np.shape(self.R)[0]
        
        #SAVE DYNAMICS
        self.dNdt, self.dRdt = dynamics
        
        #SAVE PARAMETERS
        self.params = params.copy()
        for item in self.params:#strip parameters from DataFrames if necessary
            if isinstance(self.params[item],pd.DataFrame):
                self.params[item]=self.params[item].values.squeeze()
            elif isinstance(self.params[item],list):
                self.params[item]=np.asarray(self.params[item])
        if 'D' not in params:#supply dummy values for D and e if D is not specified
            self.params['D'] = np.ones((self.M,self.M))
            self.params['e'] = 1
        
        #SAVE SCALE
        self.scale = scale
            
    def copy(self):
        return copy.deepcopy(self)

    def Reset(self,init_state):
        """
        Reset plate with new initial state, keeping same parameters.
        """
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
        
    def dydt(self,y,t,params,S_comp):
        """
        Combine N and R into a single vector with a single dynamical equation
        
        y = [N1,N2,N3...NS,R1,R2,R3...RM]
        
        t = time
        
        params = params to pass to dNdt,dRdt
        
        S_comp = number of species in compressed consumer vector
            (with extinct species removed)
        """
        return np.hstack([self.dNdt(y[:S_comp],y[S_comp:],params),
                          self.dRdt(y[:S_comp],y[S_comp:],params)])
            
    def Propagate(self,T,compress_resources=False):
        """
        Propagate the state variables forward in time according to dNdt, dRdt.
        
        T = time interval for propagation
        
        compress_resources specifies whether zero-abundance resources should be
            ignored during the propagation. This makes sense when the resources
            are non-renewable.
        """
        #CONSTRUCT FULL SYSTEM STATE
        y_in = self.N.append(self.R).T.values
        
        #PREPARE INTEGRATOR FOR PARALLEL PROCESSING
        IntegrateTheseWells = partial(IntegrateWell,self,self.params,
                                      T=T,compress_resources=compress_resources)
        
        #INITIALIZE PARALLEL POOL AND SEND EACH WELL TO ITS OWN WORKER
        pool = Pool()
        y_out = np.asarray(pool.map(IntegrateTheseWells,y_in)).squeeze().T
        pool.close()

        if len(np.shape(y_out)) == 1:
            y_out = y_out[:,np.newaxis]
        
        #UPDATE STATE VARIABLES WITH RESULTS OF INTEGRATION
        self.N = pd.DataFrame(y_out[:self.S,:],
                              index = self.N.index, columns = self.N.keys())
        self.R = pd.DataFrame(y_out[self.S:,:],
                              index = self.R.index, columns = self.R.keys())
        
    def Passage(self,f,scale=None,refresh_resource=True):
        """
        Transfer cells to a fresh plate.
        
        f = matrix specifying fraction of each old well (column) to transfer 
            to each new well (row)
            
        scale = option for using a different scale factor from the one defined 
            for the plate on initialization.
            
        refresh_resource says whether the new plate comes supplied with fresh 
            media. The resource concentrations in the media are assumed to be
            the same as the initial resource concentrations from the first plate.
            The "Reset" method can be used to adjust these concentrations.
        """
        #HOUSEKEEPING
        if scale == None:
            scale = self.scale #Use scale from initialization by default
        f = np.asarray(f) #Allow for f to be a dataframe
        self.N[self.N<0] = 0 #Remove any negative values that may have crept in
        self.R[self.R<0] = 0
        
        #DEFINE NEW VARIABLES
        N_tot = np.sum(self.N)
        R_tot = np.sum(self.R)
        N = np.zeros(np.shape(self.N))
        
        #MULTINOMIAL SAMPLING
        #(simulate transfering a finite fraction of a discrete collection of cells)
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
            R_tot = np.sum(self.R)
            R = np.zeros(np.shape(self.R))
            for k in range(self.n_wells):
                for j in range(self.n_wells):
                    if f[k,j] > 0 and R_tot[j] > 0:
                        R[:,k] += np.random.multinomial(int(scale*R_tot[j]*f[k,j]),(self.R/R_tot).values[:,j])*1./scale
            self.R = pd.DataFrame(R, index = self.R.index, columns = self.R.keys())
        
    def RunExperiment(self,f,T,np,group='Well',scale=None,refresh_resource=True):
        """
        Repeatedly propagate and passage, simulating a serial transfer experiment.
        
        f = matrix specifying fraction of each old well (column) to transfer 
            to each new well (row)
            
        T = time interval for propagation between transfers
        
        np = number of repetitions to execute
        
        group = {'Well','Species'} specifies orientation of state matrices for
            saving trajectories. Choosing 'Well' transposes the matrices before
            appending them to the trajectory DataFrame, which is usually the most 
            convenient for visualization.
            
        scale = option for using a different scale factor from the one defined 
            for the plate on initialization.
            
        refresh_resource says whether the new plate comes supplied with fresh 
            media. The resource concentrations in the media are assumed to be
            the same as the initial resource concentrations from the first plate.
            The "Reset" method can be used to adjust these concentrations.
            
        N_traj, R_traj are trajectory DataFrames. They are formed by appending
            the new system state after each propagation, using Pandas multiindex
            functionality to add a time stamp. 
        """
        if scale == None:
            scale = self.scale #Use scale from initialization by default
        t = 0
        
        #INITIALIZE TRAJECTORIES
        N_traj = TimeStamp(self.N,t,group=group)
        R_traj = TimeStamp(self.R,t,group=group)

        #PASSAGE, PROPAGATE, RECORD
        for j in range(np):
            self.Passage(f,scale=scale,refresh_resource=refresh_resource)
            self.Propagate(T)
            t += T
            N_traj = N_traj.append(TimeStamp(self.N,t,group=group))
            R_traj = R_traj.append(TimeStamp(self.R,t,group=group))
        
        return N_traj, R_traj
    
    
    def TestWell(self,T = 4,well_name = None,f0 = 1.,ns=100,log_time = False,T0=0,
                 compress_resources=False,show_plots=True):
        """
        Run a single well and plot the trajectory.
        
        T = duration of trajectory
        
        well_name = label of well to run (will choose first well if "None")
        
        f0 = fraction by which to reduce initial consumer populations. This is
            useful when running a serial transfer simulation, where the initial
            populations for the next plate will be a small fraction of the current
            values
            
        ns = number of time points to sample
        
        log_time allows one to use a logarithmic time axis, which is helpful if
            the community has very fast initial transient dynamics followed by 
            a slow convergence to steady state
            
        compress_resources specifies whether zero-abundance resources should be
            ignored during the propagation. This makes sense when the resources
            are non-renewable.
        """
        #EXTRACT STATE OF A SINGLE WELL
        if well_name == None:
            well_name = self.N.keys()[0]
        N_well = self.N.copy()[well_name] * f0
        R_well = self.R.copy()[well_name]
        
        #INTEGRATE WELL
        t, out = IntegrateWell(self,self.params,N_well.append(R_well).values,T=T,ns=ns,T0=T0,
                               return_all=True,log_time=log_time,compress_resources=compress_resources)
        
        Ntraj = out[:,:self.S]
        Rtraj = out[:,self.S:]
        
        #PLOT TRAJECTORY
        if show_plots:
            f, axs = plt.subplots(2,sharex=True)

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