#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:59:19 2017

@author: robertmarsland
"""
from community_simulator import Community, usertools, visualization
import numpy as np

init_state1, dynamics, params1 = usertools.UniformRandomCRM(main_resource_ind=0,Stot=10,Sbar=5,n_demes=3)
init_state2, dynamics, params2 = usertools.UniformRandomCRM(main_resource_ind=1,Stot=10,Sbar=5,n_demes=3)
Batch1 = Community(init_state1, dynamics, params1)
Batch2 = Batch1.copy()
Batch2.Reset(init_state2)
Batch_mix, N_1, N_2, N_sum = usertools.MixPairs(Batch1,Batch2,R0_mix = 'Com2')