#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Example of Linear Price Response Function
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import inventorize as inv
import matplotlib.pyplot as plt

price = [5, 10, 15, 17, 20, 23, 25]
demand = [600, 550, 700, 680, 500, 400, 250]

pricing_data = pd.DataFrame({'price': price, 'demand': demand})

plt.scatter(price, demand)

logit_linear = inv.single_product_optimization(x = price, 
                                y = demand, 
                                service_product_name = 'Mango',
                                current_price = 20,
                                cost = 4)

type(logit_linear)

logit_linear.keys()

predictions_Data = logit_linear['predictions']

plt.scatter(predictions_Data.x,predictions_Data.y)
plt.plot(predictions_Data.x, predictions_Data.lm_p)
plt.plot(predictions_Data.x, predictions_Data.logit_p)

logit_linear['point_of_maximum_profits']

import numpy as np
## logit or linear
retail_clean = pd.read_csv('retail_clean.csv')
retail_clean.info()
retail_clean['InvoiceDate'] = pd.to_datetime(retail_clean['InvoiceDate'])

retail_clean['year'] = retail_clean['InvoiceDate'].dt.year
retail_clean['week'] = retail_clean['InvoiceDate'].dt.week

retail_clean['weekyear'] = retail_clean['InvoiceDate'].dt.strftime('%W %Y')

weekly_sales = retail_clean.groupby(['Description', 'weekyear']).agg(total_sales=('Quantity', 'sum'),
                                                      price = ('Price', 'mean')).reset_index()

keys = weekly_sales.Description.unique()

empty_data_logit = {}

for key in keys:
    a = weekly_sales[weekly_sales.Description == key]
    cost = 0.4 * max(a['price'])
    current_price = a['price'].mean()
    logit = inv.single_product_optimization(a['price'], a['total_sales'], key, current_price, cost)
    empty_data_logit[key] = logit    


