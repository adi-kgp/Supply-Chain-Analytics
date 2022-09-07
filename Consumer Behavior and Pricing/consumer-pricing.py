#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Example of Linear Price Response Function
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import inventorize as inv

price = [5, 10, 15, 17, 20, 23, 25]
demand = [600, 550, 700, 680, 500, 400, 250]

pricing_data = pd.DataFrame({'price': price, 'demand': demand})

model = LinearRegression()
model.fit(pricing_data[['price']], pricing_data[['demand']])
model.intercept_
model.coef_

#########
guava_cost = 2.5

simulation_data = pd.DataFrame({'price': range(1,50)})
simulation_data['expected_demand'] = model.intercept_[0] + model.coef_[0]*simulation_data['price']
simulation_data['total_cost'] = simulation_data['expected_demand'] + guava_cost
simulation_data['revenue'] = simulation_data['expected_demand']*simulation_data['price']
simulation_data['profit'] = simulation_data['revenue'] - simulation_data['total_cost']
# point of maximum profit
simulation_data[simulation_data['profit'] == max(simulation_data['profit'])]

import matplotlib.pyplot as plt

plt.plot(simulation_data.price, simulation_data.expected_demand, label='demand')
plt.plot(simulation_data.price, simulation_data.revenue, label='revenue')
plt.plot(simulation_data.price, simulation_data.profit, label='profit')
plt.legend(loc = 'upper right')

#Elasticity application

inv.linear_elasticity(prices = pricing_data['price'], 
                      Sales = pricing_data['demand'], 
                      present_price = 23, 
                      cost_of_product = 2.5)

retail_clean = pd.read_csv('retail_clean.csv')
retail_clean.info()
retail_clean['InvoiceDate'] = pd.to_datetime(retail_clean['InvoiceDate'])

retail_clean['year'] = retail_clean['InvoiceDate'].dt.year
retail_clean['week'] = retail_clean['InvoiceDate'].dt.week

retail_clean['weekyear'] = retail_clean['InvoiceDate'].dt.strftime('%W %Y')

weekly_sales = retail_clean.groupby(['Description', 'weekyear']).agg(total_sales=('Quantity', 'sum'),
                                                      price = ('Price', 'mean')).reset_index()

keys = weekly_sales.Description.unique()

len(keys)

empty_data = pd.DataFrame()

for key in keys:
    try:
        a = weekly_sales[weekly_sales.Description== key]
        cost = 0.4 * max(a['price'])
        current_price = a['price'].mean()
        elasticity = inv.linear_elasticity(a['price'], a['total_sales'],current_price, cost)
        elasticity = {k:v[0] for k,v in elasticity.items()}
        data = pd.DataFrame(elasticity, index=[0])
        data['Description'] = key
        empty_data = pd.concat([empty_data, data], axis=0)
    except:
        continue
    
empty_data    

empty_data.iloc[2,:]
