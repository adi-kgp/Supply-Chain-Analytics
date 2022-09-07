#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import array
import inventorize as inv

skus = pd.read_csv('sku_distributions.csv')

apple_juice = skus[['apple_juice']]

# Different Policy Implementations (Min Q, Periodic Review, Hybrid, Base Stock)

# 1. Min Q or (S,Q) Policy
mean_apple = apple_juice.mean() 

sd_apple = apple_juice.std()

lead_time = 7

apple_sq = inv.sim_min_Q_normal(demand = apple_juice, 
                     mean = mean_apple, 
                     sd = sd_apple, 
                     leadtime = 7, 
                     service_level = 0.8, 
                     Quantity=100,
                     shortage_cost=1,
                     ordering_cost=1,
                     inventory_cost=1)

apple_sq[0].to_csv('Apple_sq.csv')

# 2. Min-max policy
grape_juice = inv.sim_min_max_pois(demand=skus.grape_juice, 
                                   lambda1=skus.grape_juice.mean(), 
                                   leadtime=7, 
                                   service_level=0.8, 
                                   Max=30,
                                   shortage_cost=1,
                                   ordering_cost=1,
                                   inventory_cost=1)

grape_juice[0].to_csv('grape_juice.csv')


## 3. Periodic policy
cantalop_juice = skus[['cantalop_juice']]

cantalop = inv.Periodic_review_pois(demand=cantalop_juice, 
                         lambda1=cantalop_juice.mean(), 
                         leadtime=7, 
                         service_level=0.9, 
                         Review_period=3,
                         ordering_cost=1,
                         inventory_cost=1,
                         shortage_cost=1)

# 4. Hybrid Policy
cantalop_hybrid = inv.Hibrid_pois(demand=cantalop_juice, 
                         lambda1=cantalop_juice.mean(), 
                         leadtime=7, 
                         service_level=0.9, 
                         Review_period=3,
                         ordering_cost=1,
                         inventory_cost=1,
                         shortage_cost=1,
                         Min=120)

cantalop_hybrid[0].to_csv('cantalop_hybrid.csv')

### base policy
apple_base = inv.sim_base_normal(demand = apple_juice, 
                     mean = mean_apple, 
                     sd = sd_apple, 
                     leadtime = 7, 
                     service_level = 0.8, 
                     shortage_cost=1,
                     ordering_cost=100,
                     inventory_cost=1)


## Comparison of five policies (with apple_juice as the reference data)
apple_sq = inv.sim_min_Q_normal(demand = apple_juice, 
                     mean = mean_apple, 
                     sd = sd_apple, 
                     leadtime = 7, 
                     service_level = 0.8, 
                     Quantity=100,
                     shortage_cost=1,
                     ordering_cost=100,
                     inventory_cost=1)

apple_base = inv.sim_base_normal(demand = apple_juice, 
                     mean = mean_apple, 
                     sd = sd_apple, 
                     leadtime = 7, 
                     service_level = 0.8, 
                     shortage_cost=1,
                     ordering_cost=100,
                     inventory_cost=1)

apple_minmax = inv.sim_min_max_normal(demand = apple_juice, 
                                      mean = mean_apple, 
                                      sd = sd_apple, 
                                      leadtime = 2, 
                                      service_level = 0.8, 
                                      Max = 400,
                                      shortage_cost = 1,
                                      ordering_cost = 100,
                                      inventory_cost = 1)

apple_periodic = inv.Periodic_review_normal(demand = apple_juice, 
                                            mean = mean_apple, 
                                            sd = sd_apple, 
                                            leadtime = 2, 
                                            service_level = 0.8, 
                                            Review_period = 4,
                                            shortage_cost = 1,
                                            ordering_cost = 100,
                                            inventory_cost = 1)

apple_hibrid = inv.Hibrid_normal(demand=apple_juice,
                         mean= mean_apple,
                         sd = sd_apple,
                         leadtime=2, 
                         service_level=0.8, 
                         Review_period=3,
                         ordering_cost=100,
                         inventory_cost=1,
                         shortage_cost=1,
                         Min=200)

import matplotlib.pyplot as plt

plt.subplot(3,2,1)
plt.plot(apple_sq[0].period[5:], apple_sq[0].demand[5:], label='demand')
plt.plot(apple_sq[0].period[5:], apple_sq[0].sales[5:], label='sales')
plt.plot(apple_sq[0].period[5:], apple_sq[0].order[5:], label='order')
plt.scatter(apple_sq[0].period[5:], apple_sq[0].inventory_level[5:], label='inventory')
plt.title('MINQ')
plt.legend(loc='upper right')

plt.subplot(3,2,2)
plt.plot(apple_base[0].period[5:], apple_base[0].demand[5:], label='demand')
plt.plot(apple_base[0].period[5:], apple_base[0].sales[5:], label='sales')
plt.plot(apple_base[0].period[5:], apple_base[0].order[5:], label='order')
plt.scatter(apple_base[0].period[5:], apple_base[0].inventory_level[5:], label='inventory')
plt.title('base')
plt.legend(loc='upper right')

plt.subplot(3,2,3)
plt.plot(apple_minmax[0].period[5:], apple_minmax[0].demand[5:], label='demand')
plt.plot(apple_minmax[0].period[5:], apple_minmax[0].sales[5:], label='sales')
plt.plot(apple_minmax[0].period[5:], apple_minmax[0].order[5:], label='order')
plt.scatter(apple_minmax[0].period[5:], apple_minmax[0].inventory_level[5:], label='inventory')
plt.title('MinMax')
plt.legend(loc='upper right')

plt.subplot(3,2,4)
plt.plot(apple_periodic[0].period[5:], apple_periodic[0].demand[5:], label='demand')
plt.plot(apple_periodic[0].period[5:], apple_periodic[0].sales[5:], label='sales')
plt.plot(apple_periodic[0].period[5:], apple_periodic[0].order[5:], label='order')
plt.scatter(apple_periodic[0].period[5:], apple_periodic[0].inventory_level[5:], label='inventory')
plt.title('Periodic')
plt.legend(loc='upper right')

plt.subplot(3,2,5)
plt.plot(apple_hibrid[0].period[5:], apple_hibrid[0].demand[5:], label='demand')
plt.plot(apple_hibrid[0].period[5:], apple_hibrid[0].sales[5:], label='sales')
plt.plot(apple_hibrid[0].period[5:], apple_hibrid[0].order[5:], label='order')
plt.scatter(apple_hibrid[0].period[5:], apple_hibrid[0].inventory_level[5:], label='inventory')
plt.title('Hibrid')
plt.legend(loc='upper right')

