#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:25:50 2017

@author: robertmarsland
"""

from community_simulator import Community,usertools
import numpy as np
import pandas as pd
import boto3
from boto3.dynamodb.conditions import Key
import json
from decimal import Decimal

def ToList(obj):
    if type(obj) == pd.DataFrame:
        return list(map(list,obj.values))
    elif type(obj) == dict:
        for item in obj:
            if type(obj[item]) == pd.DataFrame:
                obj[item] = list(map(list,obj[item].values))
            elif type(obj[item]) == np.ndarray:
                if len(np.shape(obj[item])) == 2:
                    obj[item] = list(map(list,obj[item]))
                elif len(np.shape(obj[item])) == 1:
                    obj[item] = list(obj[item])
        return obj

#CONSTRUCT DYNAMICS
assumptions = {'regulation':'independent','replenishment':'renew','response':'type I'}
def dNdt(N,R,params):
    return usertools.MakeConsumerDynamics(**assumptions)(N,R,params)
def dRdt(N,R,params):
    return usertools.MakeResourceDynamics(**assumptions)(N,R,params)
dynamics = [dNdt,dRdt]

def RunCommunity(K=500.,q=0.,e=0.2,fs=0.25,fw=0.25,food=0,Ddiv=0.2,n_types=4,c1=1,
                 MA=25,SA=40,Sgen=40,S=100,n_iter=200,T=5,n_wells=27,run_number=0,
                 param_key=None,init_keys=None,extra_time=False):
    """
    Generate communities and grow on a single externally supplied nutrient.
    
    K = chemostat set point for external nutrient supply
    
    q = family preference strength
    
    e = efficiency
    
    fs = fraction of secreted flux to same resource type
    
    fw = fraction of secreted flux to waste resource type
    
    food = index of externally supplied resource
    
    Ddiv = diversity parameter for D matrix
    
    n_types = number of food types
    
    c1 = specific consumption rate for binary sampling
    
    MA = number of resource species per type
    
    SA = number of consumer species per family
    
    Sgen = number of generalist species
    
    S = number of consumer species to initialize in each well
    
    n_iter = number of propagate/passage cycles to run
    
    T = amount of time to propagate before zeroing out low concentrations
    
    n_wells = number of wels
    
    run_number = index for keeping track of multiple runs
    
    params = model parameters (will re-sample if None)
    
    N0 = initial consumer concentrations (will re-sample if None)
    
    extra_time = add several propagation cycles of duration 10, and then 1000,
        to make sure slow modes have converged
    """
    
    #CONNECT TO DYNAMODB
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    metadata_db = dynamodb.Table('Metadata')
    final_db = dynamodb.Table('Final_State')
    initial_db = dynamodb.Table('Initial_State')
    parameters_db = dynamodb.Table('Parameters')
    data_ID = np.random.randint(0,1e10,size=n_wells)
    
    #PREPARE VARIABLES
    #Make sure MA, SA, Sgen and S are integers:
    MA = int(round(MA))
    SA = int(round(SA))
    Sgen = int(round(Sgen))
    S=int(round(S))
    M = int(MA*n_types)
    #Specify metaparameters:
    sample_par = {'SA': SA, #Number of species in each family
          'MA': MA, #Number of resources of each type
          'n_types': n_types,
          'Sgen': Sgen, #Number of generalist species
          'muc': 10, #Mean sum of consumption rates
          'q': q, #Preference strength 
          'c0':0.01, #Background consumption rate in binary model
          'c1':c1, #Maximum consumption rate in binary model
          'fs':fs, #Fraction of secretion flux with same resource type
          'fw':fw, #Fraction of secretion flux to 'waste' resource
          'D_diversity':0.2, #Variability in secretion fluxes among resources (must be less than 1)
          'waste_type':n_types-1,
          'Food':food,
          'S':S
         }
    
    #DEFINE INITIAL CONDITIONS
    S_tot = int(sample_par['SA']*n_types)+sample_par['Sgen']
    if init_keys is None:
        #Create resource vector and set food supply:
        R0 = np.zeros((M,n_wells))
        R0[food,:] = K
        
        #Sample consumers from regional pool:
        N0 = np.zeros((S_tot,n_wells))
        init_keys = np.random.randint(0,1e10,size=n_wells)
        for k in range(n_wells):
            N0[np.random.choice(S_tot,size=S,replace=False),k]=1.
            initial_db.put_item(Item={'key':init_keys[k],
                                'Consumers':json.dumps(list(N0[:,k])),
                                'Resources':json.dumps(list(R0[:,k]))})
    else:
        #Load initial conditions from database
        N0 = []
        for key in init_keys:
            response = initial_db.query(KeyConditionExpression=Key('key').eq(key))
            N0.append(json.loads(response['Items'][0]['Consumers']))
        N0 = np.asarray(N0).T
        n_wells = len(init_keys)
        R0 = np.zeros((M,n_wells))
        R0[food,:] = K

    #SAMPLE PARAMETERS
    if param_key is None:
        param_key = np.random.randint(0,1e10)
        c, D = usertools.MakeMatrices(metaparams=sample_par, kind='Binary')
        params={'c':c.values,
                'm':np.ones(S_tot)+0.1*np.random.randn(S_tot),
                'w':np.ones(M),
                'D':D.values,
                'g':np.ones(S_tot),
                'e':e,
                'r':1.,
                'tau':1
                }
        parameters_db.put_item(Item={'key':param_key,
                              'Parameters':json.dumps(ToList(params)),
                          'Resource_Names':json.dumps(list(c.keys())),
                          'Consumer_Names':json.dumps(list(c.index))})
    else:
        response = parameters_db.query(KeyConditionExpression=Key('key').eq(param_key))
        params=json.loads(response['Items'][0]['Parameters'])
        params['e'] = e
        
    #INITIALIZE COMMUNITY
    init_state = [N0,R0]
    params['R0']=R0[:,0]
    MyPlate = Community(init_state,dynamics,params)
    
    #SIMULATE
    Ntraj,Rtraj = MyPlate.RunExperiment(np.eye(n_wells),T,n_iter,refresh_resource=False,scale=1e6)
    if extra_time:
        Ntraj2,Rtraj2 = MyPlate.RunExperiment(np.eye(n_wells),100,10,refresh_resource=False,scale=1e6)
        Ntraj2,Rtraj2 = MyPlate.RunExperiment(np.eye(n_wells),1000,10,refresh_resource=False,scale=1e6)
    MyPlate.Passage(np.eye(n_wells),refresh_resource=False,scale=1e6)

    metadata = sample_par
    metadata.update(assumptions)
    for item in metadata:
        if type(metadata[item]) == float:
            metadata[item] = Decimal(str(metadata[item]))
    for k in range(n_wells):
        final_db.put_item(Item={'sample-id':data_ID[k],
                          'Consumers':json.dumps(list(MyPlate.N.values[:,k])),
                          'Resources':json.dumps(list(MyPlate.R.values[:,k]))
                          })
        metadata['sample-id'] = data_ID[k]
        metadata['Initial_State'] = init_keys[k]
        metadata['Parameters'] = param_key
        metadata_db.put_item(Item=metadata)

    return {'sample-id':data_ID,'init_key':init_keys,'param_key':param_key}
