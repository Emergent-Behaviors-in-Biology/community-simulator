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
from .essentialtools import IntegrateWell, OptimizeWell, TimeStamp

#Parameter dimensions for MicroCRM and Lotka-Volterra
dim_default = {
                'SxM':['c'],
                'MxM':['D','Di'],
                'SxS':['alpha'],
                'S':['m','g','K'],
                'M':['e','w','r','tau','R0']
                }

class Community:
    def __init__(self,init_state,dynamics,params,dimensions=dim_default,scale=10**9,parallel=True):
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
            
        params = {'name':value,...} is a Python dictionary containing names and values
            for all parameters. Parameters that are matrices or vectors (such as the
            consumer preference matrix) should have their dimensions recorded in the
            next argument. This is done automatically for the parameters of the built-
            in Microbial Consumer Resource Model, but must be done by hand for custom
            models.

        dimensions = {'SxM':[name1,name2,...],...} is a dictionary specifying the 
            dimensions of all the parameters. These are used for compressing 
            the parameter arrays when species or resources are extinct. See default
            dictionary above for proper format. Allowed dimensions are SxM, SxS, 
            MxM, M and S, where M is the number of resource types and S is the number
            of consumer species.
            
        scale is a conversion factor specifying the number of individual microbial 
            cells present when N = 1. It is used in the Passage method defined 
            below to perform multinomial sampling, and controls the strength
            of population noise. 
            
        parallel allows for disabeling parallel integration, which is currently not
            supported for Windows machines
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
        if isinstance(self.params,list): #Allow parameter file to be a list
            assert len(self.params) == self.n_wells, 'Length of parameter list must equal n_wells.'
            for k in range(len(self.params)):
                for item in self.params[k]:#strip parameters from DataFrames if necessary
                    if isinstance(self.params[k][item],pd.DataFrame):
                        if item != 'c':
                            self.params[k][item]=self.params[k][item].values.squeeze()
                        else:
                            self.params[k][item]=self.params[k][item].values
                    elif isinstance(self.params[k][item],list):
                        self.params[k][item]=np.asarray(self.params[k][item])
                    if 'D' not in self.params[k]:#supply dummy values for D and l if D is not specified
                        self.params[k]['D'] = np.ones((self.M,self.M))
                        self.params[k]['l'] = 0
                self.params[k]['S'] = self.S
        else:
            for item in self.params:#strip parameters from DataFrames if necessary
                if isinstance(self.params[item],pd.DataFrame):
                    if item != 'c':
                        self.params[item]=self.params[item].values.squeeze()
                    else:
                        self.params[item]=self.params[item].values
                elif isinstance(self.params[item],list):
                    self.params[item]=np.asarray(self.params[item])
            if 'D' not in params:#supply dummy values for D and l if D is not specified
                self.params['D'] = np.ones((self.M,self.M))
                self.params['l'] = 0
            self.params['S'] = self.S
        
        #SAVE DIMENSIONS, SCALE AND PARALLEL
        self.dimensions = dimensions
        self.scale = scale
        self.parallel = parallel
            
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
    
    def SteadyState(self,supply='external',tol=1e-7,shift_size=1,alpha=0.5,
                    eps=1e-10,R0t_0=10,max_iters=1000,verbose=False,plot=False):
        """
        Find the steady state using convex optimization.
        
        supply = {external, self-renewing}
        """
        
        #CONSTRUCT FULL SYSTEM STATE
        y_in = self.N.append(self.R).values
        
        #PACKAGE SYSTEM STATE AND PARAMETERS IN LIST OF DICTIONARIES
        if not isinstance(self.params,list):
            params = [self.params]*self.n_wells
        else:
            params = self.params
        well_info = [{'y0':y_in[:,k],'params':params[k]} for k in range(self.n_wells)]
        
        #PREPARE OPTIMIZER FOR PARALLEL PROCESSING
        OptimizeTheseWells = partial(OptimizeWell,supply=supply,tol=tol,alpha=alpha,
                                     shift_size=shift_size,max_iters=max_iters,
                                     eps=eps,R0t_0=R0t_0,verbose=verbose,dimensions=self.dimensions)
        
        if self.parallel:
            #INITIALIZE PARALLEL POOL AND SEND EACH WELL TO ITS OWN WORKER
            pool = Pool()
            y_out = np.asarray(pool.map(OptimizeTheseWells,well_info)).squeeze().T
            pool.close()
        else:
            #IF PARALLEL IS DEACTIVATED, USE ORDINARY MAP
            y_out = np.asarray(list(map(OptimizeTheseWells,well_info))).squeeze().T
        if len(np.shape(y_out)) == 1:#handle case of single-well plate
            y_out = y_out[:,np.newaxis]
        
        #UPDATE STATE VARIABLES WITH RESULTS OF OPTIMIZATION
        self.N = pd.DataFrame(y_out[:self.S,:],
                              index = self.N.index, columns = self.N.keys())
        self.R = pd.DataFrame(y_out[self.S:,:],
                              index = self.R.index, columns = self.R.keys())

        #PRINT DIAGNOSTICS
        dNdt_f = np.asarray(list(map(self.dNdt,self.N.T.values,self.R.T.values,params)))
        dRdt_f = np.asarray(list(map(self.dRdt,self.N.T.values,self.R.T.values,params)))
        
        if plot:
            dNdt_f = np.asarray(list(map(self.dNdt,self.N.T.values,self.R.T.values,params))).reshape(-1)
            dRdt_f = np.asarray(list(map(self.dRdt,self.N.T.values,self.R.T.values,params))).reshape(-1)
            N = self.N.values.reshape(-1)
            R = self.R.values.reshape(-1)
    
            fig,ax = plt.subplots()
            ax.plot(dNdt_f[N>0]/N[N>0],'o',markersize=1)
            ax.set_ylabel('Per-Capita Growth Rate')
            ax.set_title('Consumers')
            plt.show()
            
            fig,ax = plt.subplots()
            ax.plot(dRdt_f/R,'o',markersize=1)
            ax.set_ylabel('Per-Capita Growth Rate')
            ax.set_title('Resources')
            plt.show()
            
            
    def Propagate(self,T,compress_resources=False,compress_species=True):
        """
        Propagate the state variables forward in time according to dNdt, dRdt.
        
        T = time interval for propagation
        
        compress_resources specifies whether zero-abundance resources should be
            ignored during the propagation. This makes sense when the resources
            are non-renewable.
            
        compress_species specifies whether zero-abundance species should be
            ignored during the propagation. This always makes sense for the
            models we consider. But for user-defined models with new parameter
            names, it must be turned off, since the package does not know how
            to compress the parameter matrices properly.
        """
        #CONSTRUCT FULL SYSTEM STATE
        y_in = self.N.append(self.R).values
        
        #PACKAGE SYSTEM STATE AND PARAMETERS IN LIST OF DICTIONARIES
        if isinstance(self.params,list):
            well_info = [{'y0':y_in[:,k],'params':self.params[k]} for k in range(self.n_wells)]
        else:
            well_info = [{'y0':y_in[:,k],'params':self.params} for k in range(self.n_wells)]
        
        #PREPARE INTEGRATOR FOR PARALLEL PROCESSING
        IntegrateTheseWells = partial(IntegrateWell,self,T=T,compress_resources=compress_resources,
                                      compress_species=compress_species)
        
        #INITIALIZE PARALLEL POOL AND SEND EACH WELL TO ITS OWN WORKER
        if self.parallel:
            pool = Pool()
            y_out = np.asarray(pool.map(IntegrateTheseWells,well_info)).squeeze().T
            pool.close()
        else:
            y_out = np.asarray(list(map(IntegrateTheseWells,well_info))).squeeze().T

        #HANDLE CASE OF A SINGLE-WELL PLATE
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
        
    def RunExperiment(self,f,T,npass,group='Well',scale=None,refresh_resource=True,
                      compress_resources=False,compress_species=True):
        """
        Repeatedly propagate and passage, simulating a serial transfer experiment.
        
        f = matrix specifying fraction of each old well (column) to transfer 
            to each new well (row)
            
        T = time interval for propagation between transfers
        
        npass = number of repetitions to execute
        
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
            
        compress_resources specifies whether zero-abundance resources should be
            ignored during the propagation. This makes sense when the resources
            are non-renewable.
            
        compress_species specifies whether zero-abundance species should be
            ignored during the propagation. This always makes sense for the
            models we consider. But for user-defined models with new parameter
            names, it must be turned off, since the package does not know how
            to compress the parameter matrices properly.
            
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
        for j in range(npass):
            self.Passage(f,scale=scale,refresh_resource=refresh_resource)
            self.Propagate(T,compress_resources=compress_resources,compress_species=compress_species)
            t += T
            N_traj = N_traj.append(TimeStamp(self.N,t,group=group))
            R_traj = R_traj.append(TimeStamp(self.R,t,group=group))
        
        return N_traj, R_traj
    
    
    def TestWell(self,T = 4,well_name = None,f0 = 1.,ns=100,log_time = False,T0=0,
                 compress_resources=False,compress_species=False,show_plots=True,axs=[]):
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
        if isinstance(self.params,list):
            params_well = self.params[np.where(np.asarray(self.N.keys())==well_name)[0][0]]
        else:
            params_well = self.params
        
        #INTEGRATE WELL
        t, out = IntegrateWell(self,{'y0':N_well.append(R_well).values,'params':params_well},T=T,ns=ns,T0=T0,
                               return_all=True,log_time=log_time,compress_resources=compress_resources,
                               compress_species=compress_species)
        
        Ntraj = out[:,:self.S]
        Rtraj = out[:,self.S:]
        
        #PLOT TRAJECTORY
        if show_plots:
            if axs == []:
                fig, axs = plt.subplots(2,sharex=True)
            else:
                assert len(axs) == 2, 'Must supply two sets of axes.'

            if log_time:
                axs[0].semilogx(t,Ntraj)
                axs[1].semilogx(t,Rtraj)
            else:
                axs[0].plot(t,Ntraj)
                axs[1].plot(t,Rtraj)
            axs[0].set_ylabel('Consumer Abundance')
            axs[1].set_ylabel('Resource Abundance')
            axs[1].set_xlabel('Time')

        return t, Ntraj, Rtraj
