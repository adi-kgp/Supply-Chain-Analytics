#!/usr/bin/env python3
# -*- coding: utf-8 -*-
mean = 50
sd = 10
c = 1.2
Salvage = 0.7
penalty_term = 0.4
price = 3
cost = 1.2

import inventorize as inv

inv.CriticalRatio(sellingprice = price, 
                  cost = c, 
                  salvage = 0, 
                  penality = 0)

# using this formula in the method above: (price - c)/(price - cost + c)

# without penalty_term and salvage
inv.MPN_singleperiod(mean = mean, 
                     standerddeviation = sd, 
                     p = price, 
                     c = cost, 
                     g = 0, 
                     b = 0)

# with salvage
inv.MPN_singleperiod(mean = mean, 
                     standerddeviation = sd, 
                     p = price, 
                     c = c, 
                     g = Salvage, 
                     b = 0)

# with salvage and penalty_term
inv.MPN_singleperiod(mean = mean, 
                     standerddeviation = sd, 
                     p = price, 
                     c = c, 
                     g = Salvage, 
                     b = penalty_term)

inv.EPN_singleperiod(quantity = 40, 
                     mean = 40, 
                     standerddeviation = 10, 
                     p = price, 
                     c = c, 
                     g = 0, 
                     b = 0)
