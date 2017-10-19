#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 11:05:55 2017

@author: robertmarsland
"""

import pandas as pd
import numpy as np
from scipy import integrate

def IntegrateDeme(CommunityInstance,y0,T=1,ns=2,return_all=False):
    t = np.linspace(0,T,ns)
    if return_all:
        return t, integrate.odeint(CommunityInstance.dydt,y0,t)
    else:    
        return integrate.odeint(CommunityInstance.dydt,y0,t)[-1]
    
def TimeStamp(data,t,group='Deme'):
    if group == 'Deme':
        data_time = data.copy().T
        mdx = pd.MultiIndex.from_product([[t],data_time.index],names=['Time','Deme'])
    elif group == 'Species':    
        data_time = data.copy()
        mdx = pd.MultiIndex.from_product([[t],data.index],names=['Time','Species'])
    else:
        return 'Invalid group choice'
        
    data_time.index = mdx
    return data_time