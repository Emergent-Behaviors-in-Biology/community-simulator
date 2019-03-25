#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:09:38 2017

@author: robertmarsland
"""

import numpy as np
import matplotlib.pyplot as plt

def NonzeroColumns(data,thresh=0):
    return data.keys()[np.where(np.sum(data)>thresh)]

def StackPlot(df,ax=None,labels=False,title=None,drop_zero=True,unique_color=False,random_color=True):
    if ax == None:
        fig, ax = plt.subplots(1)
    w = len(df.keys())
    
    if drop_zero:
        dfmax = max(df.values.reshape(-1))
        df = df.loc[(df>0.01*dfmax).any(1)]
    
    if unique_color:
        if random_color:
            color_list = plt.cm.get_cmap('cubehelix')(np.random.choice(np.arange(256),size=len(df),replace=False))
        else:
            color_list = plt.cm.get_cmap('cubehelix')(np.asarray(np.linspace(0,256,len(df)),dtype=int))
        ax.stackplot(range(w),df,colors = color_list)
    else:
        ax.stackplot(range(w),df)
        
    ax.set_yticks(())
    ax.set_xticks(range(w))
    ax.set_xlim((0,w-1))
    if np.max(df.sum()) > 0:
        ax.set_ylim((0,np.max(df.sum())))
    
    if labels:
        ax.set_xticklabels((df.keys()))
    else:
        ax.set_xticks(())
        
    if title != None:
        ax.set_title(title)
    
    return ax

def PlotTraj(traj_in, dropzeros = False, plottype = 'stack', wellchoice = None,
             figsize = (10,20)):
    traj = traj_in.copy()
    if wellchoice!= None:
        for item in wellchoice:
            assert item in traj.index.levels[-1], "wellchoice must be a list of well labels"
        traj = traj.reindex(wellchoice,level=1)
        
    group = traj.index.names[-1]
    nplots = len(traj.index.levels[-1])
    f, axs = plt.subplots(nplots, sharex = True, figsize = figsize)
    k = 0
    if type(axs) not in [list,np.ndarray]:
        axs = [axs]

    for item in traj.index.levels[-1]:
            plot_data = traj.xs(item,level=group)
            if dropzeros:
                plot_data = plot_data[NonzeroColumns(plot_data)]
            if plottype == 'stack':
                StackPlot(plot_data.T,ax=axs[k])
                if k == nplots-1:
                    t_axis = traj.index.levels[0]
                    axs[k].set_xticks(range(len(t_axis)))
                    axs[k].set_xticklabels(range(len(t_axis)))
                    axs[k].set_xlabel('Dilution Cycles')
            elif plottype == 'line':
                plot_data.plot(ax = axs[k], legend = False)
            else:
                return 'Invalid plot type.'
            k+=1
