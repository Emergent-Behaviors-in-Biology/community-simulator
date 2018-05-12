#!/usr/bin/env python3
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
from matplotlib.backends import backend_pdf as bpdf

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

def RankAbundance(df,metadata,params,thresh=1e-6,title=None,fs=18,ax=None,palette=None):
    if ax is None:
        fig,ax=plt.subplots(figsize=(4,4))
        fig.subplots_adjust(bottom=0.2,left=0.2)
    data = df.copy()
    for item in params:
        data = data.loc[metadata[item]==params[item]]
    data = data.T/data.T.sum()
    data[data<thresh]=0
    
    richness = (data>0).sum()
    rmin = np.min(richness)
    rmax = np.max(richness)
    n_colors = (rmax-rmin)+1
    if palette is None:
        colors = sns.color_palette("Blues_d",n_colors)
    else:
        colors = sns.color_palette(palette,n_colors)
    
    for item in data.keys():
        ra = data[item].sort_values(ascending=False).values
        ra = ra[ra>0]
        ax.semilogy(ra,'o-',color=colors[len(ra)-rmin])
    ax.set_xlabel('Rank',fontsize=fs)
    ax.set_ylabel('Relative Abundance',fontsize=fs)
    if title is not None:
        ax.set_title(title,fontsize=fs+4)

    return ax

def Histogram(df,metadata,params,thresh=0,title=None,fs=18,ax=None,nbins=10,
              minbin=1e-4,maxbin=1,scale=1,log=False):
    if log:
        bins = 10**(np.linspace(np.log10(minbin),np.log10(maxbin),nbins))
    else:
        bins = np.linspace(0,maxbin,nbins)

    if ax is None:
        fig,ax=plt.subplots(figsize=(4,4))
        fig.subplots_adjust(bottom=0.2,left=0.2)
    data = df.copy()/scale
    for item in params:
        data = data.loc[metadata[item]==params[item]]
    data[data<thresh]=0
    
    data = data.values.reshape(-1)
    #data = data[data>0]

    ax.hist(data,bins=bins,normed=True)
    ax.set_xlabel('Abundance',fontsize=fs)
    ax.set_ylabel('Frequency',fontsize=fs)
    if log:
        ax.set_xscale('log')
    if title is not None:
        ax.set_title(title,fontsize=fs+4)

    return ax

def CompositionPlot(data,n_wells=10,PCA_examples=False,drop_zero=False,thresh=1e-6,title='test'):
    if drop_zero:
        data = data.loc[(data.T>thresh).any()]
    def_colors = sns.color_palette("RdBu_r",len(data))
    well_colors = sns.color_palette("RdBu_r",n_wells)
    PCA_model = PCA(n_components=2).fit(data.T)
    explained_variance = np.around(100*PCA_model.explained_variance_ratio_,decimals=1)
    N_PCA = PCA_model.transform(data.T)
    
    fig,axs=plt.subplots(2,figsize=(5,8))
    fig.subplots_adjust(hspace=0.3,left=0.2)
    axs[1].scatter(N_PCA[:,0],N_PCA[:,1],marker='.',color='gray')
        
    axs[1].set_xlabel('PCA 1 ('+str(explained_variance[0])+' %)',fontsize=14)
    axs[1].set_ylabel('PCA 2 ('+str(explained_variance[1])+' %)',fontsize=14)

    names = data.keys()[:n_wells]
    dominant = np.argmax(data.values,axis=0)[:n_wells]
    dominant_idx = list(set(dominant))
    names = np.array(list(zip(names,dominant)),dtype=[('well','S30'),('dominant',int)])
    names_sort = np.asarray(np.sort(names,order='dominant')['well'],dtype=str)

    f = data[names_sort]/data[names_sort].sum()
    
    f.copy().T.plot.bar(stacked=True,legend=False,ax=axs[0],color = def_colors)
    axs[0].set_xticks(())

    if PCA_examples:
        for well in range(n_wells):
            axs[1].plot(N_PCA[well,0],N_PCA[well,1],marker='o',color = well_colors[well])
            axs[0].plot(np.where(names_sort==data.keys()[well])[0][0],1.1,marker='o',color = well_colors[well])
    
    axs[0].set_title(title,fontsize=18)
    axs[0].set_ylabel('Composition',fontsize=14)
    axs[0].set_xlabel('Community',fontsize=14)

    pdf = bpdf.PdfPages('../Plots/PCA_'+title+'.pdf')
    pdf.savefig(fig)
    pdf.close()

    plt.show()
