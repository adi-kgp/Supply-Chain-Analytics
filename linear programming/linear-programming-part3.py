#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 18:21:53 2022

@author: arch_camen
"""
import pandas as pd
from pulp import *

param = pd.read_excel('Production_scheduling.xlsx')

param = param.rename(columns={'Unnamed: 0': 'period'})

param['t'] = range(1,13)

param = param.set_index('t')

inventory = LpVariable.dicts('inv', [0,1,2,3,4,5,6,7,8,9,10,11,12],0,None,'Integer')

inventory[0] = 200

production = LpVariable.dicts('Prod', [1,2,3,4,5,6,7,8,9,10,11,12],0,None,'Integer')

binary = LpVariable.dicts('binary', [1,2,3,4,5,6,7,8,9,10,11,12],0,None,'Binary')

time = [1,2,3,4,5,6,7,8,9,10,11,12]

model = LpProblem('Production', LpMinimize)

model += lpSum([inventory[t]* param.loc[t,'storage cost'] + production[t]*param.loc[t,'var'] + 
                binary[t]* param.loc[t, 'fixed cost'] for t in time])

for t in time:
    model += production[t] - inventory[t] + inventory[t-1] >= param.loc[t, 'demand']
    model += production[t] <=  binary[t] * param.loc[t, 'Capacity']
    
model.solve()

for i in production:
    print(production[i],production[i].varValue)

for v in model.variables():
    print(v, v.varValue)

