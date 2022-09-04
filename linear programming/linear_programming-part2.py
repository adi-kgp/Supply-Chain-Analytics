#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 13:53:18 2022

@author: arch_camen
"""
# Solution to assignment problem 1 (Maximization problem)
from pulp import *

model = LpProblem('PILLOWS', LpMaximize)

X1 = LpVariable('X1', 0, None, 'Integer')
X2 = LpVariable('X2', 0, None, 'Integer')
X3 = LpVariable('X3', 0, None, 'Integer')

# define our objective function
model += X1*33 + X2*40 + X3*34

model += X1*0.4 + X2*0.7 + X3*0.4 <=40
model += X1*0.2 + X2*0.5 + X3*0.6 <=40
model += X1*0.3 + X2*0.3 + X3*0.2 <=40

model.solve()

X1.varValue
X2.varValue
X3.varValue

# Optimization Problem 2 (minimization problem)
from pulp import *

model = LpProblem('shipping', LpMinimize)

customers = ['Australia', 'Sweden', 'Brazil']
factory = ['Factory1', 'Factory2']
products = ['Chair', 'Table', 'Beds']

keys = [(f,p,c) for f in factory for p in products for c in customers]

var = LpVariable.dicts('shipment', keys, 0, None, cat = 'Integer')

costs_value = [50,80,50,
               60,90,60,
               70,90,70,
               80,50,80,
               90,60,90,
               90,70,90]

costs = dict(zip(keys, costs_value))

demand_keys = [(p,c) for c in customers for p in products]

demand_values = [50,80,200,
                 120,80,40,
                 30,60,175]

demand = dict(zip(demand_keys, demand_values))

# Defining our objective function
model += lpSum(var[(f,p,c)]*costs[(f,p,c)] 
               for f in factory for p in products for c in customers)

# constraints

model += lpSum(var[('Factory1',p,c)] for p in products for c in customers) <= 500 

model += lpSum(var[('Factory2',p,c)] for p in products for c in customers) <= 500 

for c in customers:
    for p in products:
        model += var[('Factory1', p,c)] + var[('Factory2', p,c)] >= demand[(p,c)]

model.solve()

for i in var:
    print("{} shipping {}".format(i, var[i].varValue))
    


