#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import inventorize3 as inv

retail = pd.read_csv('Uk_Ireland.csv')

retail = retail.drop_duplicates()
retail = retail.dropna(axis=0)

retail.info()

retail.InvoiceDate

retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])

retail['date'] = retail.InvoiceDate.dt.strftime('%Y-%m-%d')

retail['date'] = pd.to_datetime(retail['date'])

max_date = max(retail.date)

last_three = retail[retail.date > '2011-09-01']

last_three.columns

last_three['revenue'] = last_three['Quantity'] * last_three['Price']

a = last_three.groupby(['date', 'Description']).agg(total_daily=('Quantity',np.sum),
                                                    total_revenue=('revenue', np.sum)).reset_index()

grouped = a.groupby('Description').agg(average=('total_daily', np.mean),
                                       sd = ('total_daily', 'std'),
                                       total_sales=('total_daily', np.sum),
                                       total_revenue = ('total_revenue', np.sum)).reset_index()

for_abc = inv.productmix(skus=grouped['Description'], sales=grouped['total_sales'], 
                         revenue=grouped['total_revenue'])

for_abc.product_mix.value_counts() 
""" Note:  A-C, A-B are volume drivers (high sales low margin),
  C-A, B-A (low sales high margin)
  In general: left side alphabet: fast moving or high sales item (A => B => C)
              right side alphabet: high margin item (A => B => C)
"""

lead_time = 21
sd_leadtime = 2

mapping = {
        'A_A':0.8, "A_C":0.7, "C_A": 0.8, "A_B":0.8,
        "B_A":0.8, "B_C": 0.6, "C_C":0.6, "B_B":0.7, "C_B": 0.6
    }

for_abc['service_level'] = for_abc.product_mix.map(mapping)

## reorder point

abcd = for_abc[['skus', 'service_level']]

for_reorder = pd.merge(grouped, abcd, how='left', left_on='Description', right_on='skus')

for_reorder.columns

### with leadtime variability

empty_data_ltv = pd.DataFrame()

for i in range(for_reorder.shape[0]):
    ordering_point = inv.reorderpoint_leadtime_variability(dailydemand = int(for_reorder.loc[i, 'average']),
                                                           dailystandarddeviation=for_reorder.loc[i,'sd'],
                                                           leadtimein_days=21, 
                                                           sd_leadtime_days=2, 
                                                           csl=for_reorder.loc[i,'service_level'])
    as_data = pd.DataFrame(ordering_point, index=[0])
    as_data['Description'] = for_reorder.loc[i, 'Description']
    empty_data_ltv = pd.concat([empty_data_ltv, as_data], axis=0)

empty_data_ltv        

## joining all
all_data = pd.merge(for_reorder, empty_data_ltv, how='left')

all_data['safety_stock'] = all_data['reorder_point'] - all_data['demandleadtime']

# Visualizing all_data
import seaborn as sns
all_data[all_data.safety_stock == max(all_data.safety_stock)]

all_data_modified = all_data[all_data.safety_stock != max(all_data.safety_stock)]

sns.scatterplot(x='sd', y='safety_stock', hue='service_level', data= all_data_modified)
