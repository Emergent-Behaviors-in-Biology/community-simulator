from community_simulator.usertools import MakeConsumerDynamics,MakeResourceDynamics,AddLabels,MakeMatrices,MakeInitialState
from community_simulator import Community
from community_simulator.analysis import validate_simulation
import pandas as pd
import numpy as np
import time

folder = '/project/biophys/microbial_crm/data/'

mp = {'sampling':'Binary', #Sampling method
              'MA': 100, #Number of resources of each type
              'Sgen': 0, #Number of generalist species
              'muc': 10, #Mean sum of consumption rates
              'q': 0, #Preference strength (0 for generalist and 1 for specialist)
              'c0':0.0, #Background consumption rate in binary model
              'c1':1., #Specific consumption rate in binary model
              'fs':0.45, #Fraction of secretion flux with same resource type
              'fw':0.45, #Fraction of secretion flux to 'waste' resource
              'D_diversity':0.2, #Variability in secretion fluxes among resources (must be less than 1)
              'regulation':'independent',
              'replenishment':'external',
              'response':'type I',       
              'n_wells':10, #Number of independent wells
              'food':0
             }

#Construct dynamics
def dNdt(N,R,params):
    return MakeConsumerDynamics(mp)(N,R,params)
def dRdt(N,R,params):
    return MakeResourceDynamics(mp)(N,R,params)
dynamics = [dNdt,dRdt]

data_int = pd.DataFrame(index = 10*np.arange(100,dtype=int),
                       columns = ['Run Time','Mean Accuracy','Std. Dev. Accuracy','Failures','Invasions'])
data_opt = data_int.copy()

for R0 in data_int.index:
    mp.update({'MA':100,
               'SA':200,
               'R0_food':R0,
               'S':100})

    c, D = MakeMatrices(mp)
    init_state = MakeInitialState(mp)

    m = 1+np.random.randn(200)*0.01
    #Create parameter set
    params={'c':c,
            'm':m,
            'w':1,
            'D':D,
            'g':1,
            'l':0.8,
            'R0':init_state[1][:,0],
            'tau':1
           }

    MyPlate_opt = Community(init_state,dynamics,params)

    start_time = time.time()
    out = MyPlate_opt.SteadyState()
    end_time = time.time()
    MyPlate_opt.N[MyPlate_opt.N<1e-4] = 0
    data_opt.loc[R0] = validate_simulation(MyPlate_opt,init_state[0])
    data_opt.loc[R0,'Run Time']= end_time-start_time
    
    data_opt.to_excel(folder+'ConvexOptimization_R0_sweep.xlsx')
