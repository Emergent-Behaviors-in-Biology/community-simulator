#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:09:38 2017

@author: robertmarsland
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import k_means

def NonzeroColumns(data,thresh=0):
    return data.keys()[np.where(np.sum(data)>thresh)]

def StackPlot(df,ax=None,labels=False,title=None,cluster=False,drop_zero=True,unique_color=False,random_color=True):
    if ax == None:
        fig, ax = plt.subplots(1)
    w = len(df.keys())
    
    if cluster:
        z = hierarchy.linkage(df.T,optimal_ordering=True,metric='cosine')
        idx_sort=z[:,:2].reshape(-1)
        idx_sort=np.asarray(idx_sort[idx_sort<w],dtype=int)
        df = df[df.keys()[idx_sort]]
    
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

def PlotTraj(traj_in, dropzeros = False, plottype = 'stack', demechoice = None,
             figsize = (10,20)):
    traj = traj_in.copy()
    if demechoice!= None:
        for item in demechoice:
            assert item in traj.index.levels[-1], "demechoice must be a list of deme labels"
        traj = traj.reindex(demechoice,level=1)
        
    group = traj.index.names[-1]
    nplots = len(traj.index.levels[-1])
    f, axs = plt.subplots(nplots, sharex = True, figsize = figsize)
    k = 0
    

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
            
def ReduceDimension(N,R,c,plate=0,clusters=2,perplexity=5,plot_clusters=False,plot_scatter=True):

    metagenome = c.loc[plate].T.dot(N.loc[plate])

    N_PCA = PCA(n_components=2).fit_transform(N.loc[plate].T)
    N_TSNE = TSNE(perplexity=perplexity).fit_transform(N.loc[plate].T)
    R_PCA = PCA(n_components=2).fit_transform(R.loc[plate].T)
    R_TSNE = TSNE(perplexity=perplexity).fit_transform(R.loc[plate].T)
    M_PCA = PCA(n_components=2).fit_transform(metagenome.T)
    M_TSNE = TSNE(perplexity=perplexity).fit_transform(metagenome.T)

    _, y,_ = k_means(metagenome.T, clusters)
    y_unique = np.unique(y)
    
    if plot_scatter:
        fig,axs=plt.subplots(2,3,figsize=(15,10))
        for yu in y_unique:
            pos = (y == yu)
            axs[0,0].scatter(N_PCA[pos,0],N_PCA[pos,1],label=yu)
            axs[1,0].scatter(N_TSNE[pos,0],N_TSNE[pos,1],label=yu)
            axs[0,1].scatter(R_PCA[pos,0],R_PCA[pos,1],label=yu)
            axs[1,1].scatter(R_TSNE[pos,0],R_TSNE[pos,1],label=yu)
            axs[0,2].scatter(M_PCA[pos,0],M_PCA[pos,1],label=yu)
            axs[1,2].scatter(M_TSNE[pos,0],M_TSNE[pos,1],label=yu)
        axs[0,0].set_title('OTU PCA')
        axs[1,0].set_title('OTU t-SNE')
        axs[0,1].set_title('Resource PCA')
        axs[1,1].set_title('Resource t-SNE')
        axs[0,2].set_title('Metagenome PCA')
        axs[1,2].set_title('Metagenome t-SNE')
        plt.show()
    
    if plot_clusters:
        n_cluster=max(y)+1

        fig_OTU, axs_OTU = plt.subplots(1,n_cluster,figsize=(15,5),sharey=True)
        k=0
        for yu in y_unique:
            pos = (y == yu)
            StackPlot(N.loc[plate][N.keys()[pos]],cluster=True,unique_color=True,ax=axs_OTU[k])
            k+=1

        fig_fam, axs_fam = plt.subplots(1,n_cluster,figsize=(15,5),sharey=True)
        k=0
        for yu in y_unique:
            pos = (y == yu)
            StackPlot(N.loc[plate][N.keys()[pos]].groupby(level=0).sum(),cluster=True,
                      unique_color=True,random_color=False,ax=axs_fam[k])
            k+=1
       
    return y, [N_PCA, N_TSNE, R_PCA, R_TSNE, M_PCA, M_TSNE], [fig_OTU, axs_OTU, fig_fam, axs_fam]