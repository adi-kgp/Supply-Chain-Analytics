#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
I am the supply chain manager of Coffee co, a well known brand for colombian coffee blend.
Coffee co takes raw coffee from its partner supplier in Columbia. The annual demand of raw coffee
is 4000 tons per year, the price of one ton from the supplier is 2500 USD. 

Q1: Assume a holding rate of 10% while the cost of transportation and ordering is 6000 USD, 
what should be the optimal Q? 

Q2: What is the total logistics cost? 

Q3: What is my t practical? 

Q4: If the supplier will offer you a 10%
discount if your Q is 500, would I accept it?

"""

import inventorize3 as inv
import pandas as pd

#parameters
d = 4000
c = 2500
s = 6000
h=0.1

eoq = pd.DataFrame(inv.eoq(d,s,c,h), index=[0])

eoq1 = eoq.loc[0,'EOQ'] # answer 1

### Total logistics cost TLC: d/q* s + q/2*(h*p) + D*c

TLC = (d/eoq1)*s + (eoq1/2)*(h*c) + c*d # answer 2

TQpractical = inv.TQpractical(annualdemand=d, orderingcost=s, purchasecost=c, holdingrate=h)

Qpractical = TQpractical['Qpractical'] # answer 3

## At 10% discount 

TLC1 = (d/500)*s + (500/2)*(h*c*0.9) + (c*0.9*d) # answer 4

"""Scenario 1: the lead time it takes for the orders to arrive is one month, what 
will be the reorder point?"""

t = eoq1/d

L = 1/12

L < t

reorderpoint = L * d

# Whatever our inventory level/ inventory position , we order Q and Q in this case is 438

"""Scenario 2: the lead time it takes for the orders to arrive is two months, what 
will be the reorder point?"""

L2 = 2/12

L2 < t

l_prime = L2 - (1*(eoq1/d))

reorderpoint2 = l_prime*d

## MIN Q