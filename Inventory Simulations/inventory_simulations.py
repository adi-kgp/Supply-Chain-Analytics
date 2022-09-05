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
