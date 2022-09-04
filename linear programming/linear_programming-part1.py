# -*- coding: utf-8 -*-
from pulp import *

product1 = 25
product2 = 35

## 25 kg of feather and 35 kg of cotton

## It takes 0.3 kg feather and 0.5 kg cotton to make x1
## It takes 0.5 kg feather and 0.5 kg cotton to make x2

model = LpProblem('PILLOWS', LpMaximize)

X1 = LpVariable('X1', 0, None, 'Integer')
X2 = LpVariable('X2', 0, None, 'Integer')


## Define our objective function
model += X1 * 25 + X2*35

# Define constraints 
model += X1 * 0.3 + X2 * 0.5 <= 20
model += X1 * 0.5 + X2 * 0.5 <= 35

model.solve()

# Getting the decision variables
X1.varValue
X2.varValue


