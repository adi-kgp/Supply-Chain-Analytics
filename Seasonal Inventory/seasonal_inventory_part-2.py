#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

retail_clean = pd.read_csv('retail_clean.csv')

# preparing the data for MPN
retail_clean.head()
retail_clean.columns

retail_clean.info()

retail_clean['InvoiceDate'] = pd.to_datetime(retail_clean['InvoiceDate'])

retail_clean['date'] = retail_clean['InvoiceDate'].dt.strftime('%Y-%m-%d')

retail_clean['date'] = pd.to_datetime(retail_clean['date'])

retail_clean['year'] = retail_clean['date'].dt.year

years_2 = retail_clean[retail_clean.year.isin([2010, 2011])]

total = years_2.groupby(['year','Description']).agg(total_sales=('Quantity', np.sum),
                                            price=('Price', 'mean')).reset_index()

expected = total.groupby('Description').agg(expected_Demand = ('total_sales', np.mean),
                                            sd = ('total_sales', 'std'),
                                            price = ('price', np.mean)).reset_index()

expected.head()

# margin of error

def margin_error(dataframe):
    if(pd.isna(dataframe['sd'])):
        a = dataframe['expected_Demand']*0.1
    else:
        a = dataframe['sd']
    return a

expected['sd1'] = expected.apply(margin_error, axis=1)    

expected['cost'] = expected['price']*0.6 

empty_data = pd.DataFrame()

for i in range(expected.shape[0]):
    a = inv.MPN_singleperiod(mean=expected.loc[i,'expected_Demand'],
                             standerddeviation = expected.loc[i,'sd1'],
                             p = expected.loc[i,'price'],
                             c = expected.loc[i,'cost'],
                             g = 0,
                             b = 0)
    b = pd.DataFrame(a, index=[0])
    b['description'] = expected.loc[i, 'Description']
    empty_data = pd.concat([empty_data, b], axis=0)
    
empty_data.head()

# example inspection

empty_data.iloc[1,:]
