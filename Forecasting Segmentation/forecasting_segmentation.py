# -*- coding: utf-8 -*-
import pandas as pd

retail = pd.read_csv('retail_clean.csv')

retail.info()

retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])

retail['daysofweek'] = retail['InvoiceDate'].dt.dayofweek

retail['daysofweek'].value_counts()

retail['date'] = retail['InvoiceDate'].dt.strftime('%Y-%m-%d')

### CV2 (Coeeficient of Variance) and the average demand interval
retail_grouped = retail.groupby(['Description','date']).agg(total_sales=('Quantity','sum')).reset_index()

cv_data = retail_grouped.groupby(['Description']).agg(average=('total_sales', 'mean'), 
                                                      sd = ('total_sales', 'std')).reset_index()

cv_data['cv_squared'] = (cv_data['sd']/cv_data['average'])**2

## The average demand interval per product

product_by_date = retail.groupby(['Description','date']).agg(count=('Description', 'count')).reset_index()

skus = product_by_date.Description.unique()

empty_df = pd.DataFrame()

""" due to hardware limitations, this code is repeated five times instead of a single 
 execution ( from 0 to 1000, 1000 to 2000 and so on till len(skus))
for i in range(5000, len(skus)):
    a = product_by_date[product_by_date.Description==skus[i]]
    a['previous_date'] = a['date'].shift(1)
    empty_df = pd.concat([empty_df, a], axis=0)
    

with open('empty_df.csv', 'w') as csv_file:
    empty_df.to_csv(path_or_buf=csv_file)
"""

empty_dataframe = pd.read_csv('empty_df.csv')
empty_dataframe.drop('Unnamed: 0', axis=1, inplace=True)
empty_dataframe['date'] = pd.to_datetime(empty_dataframe.date)
empty_dataframe['previous_date'] = pd.to_datetime(empty_dataframe.previous_date)

empty_dataframe['Duration'] = empty_dataframe['date'] - empty_dataframe['previous_date']
empty_dataframe['duration'] = empty_dataframe['Duration'].astype('string').str.replace('days', '')
empty_dataframe['duration'] = pd.to_numeric(empty_dataframe['duration'])

## Calculating Average Demand Interval (ADI)
ADI = empty_dataframe.groupby('Description').agg(ADI = ('duration', 'mean')).reset_index()

adi_cv = pd.merge(ADI, cv_data)

def category(dataframe):
    b = 0
    if ((dataframe['ADI']<=1.34) & (dataframe['cv_squared']<=0.49)):
        b = 'smooth'
    if ((dataframe['ADI']>=1.34) & (dataframe['cv_squared']>=0.49)):
        b = "Lumpy"
    if ((dataframe['ADI']<1.34) & (dataframe['cv_squared']>0.49)):
        b = 'Erratic'
    if ((dataframe['ADI']>1.34) & (dataframe['cv_squared']<0.49)):
        b = 'Intermittent'
    return b

adi_cv['category'] = adi_cv.apply(category, axis=1)

import seaborn as sns

sns.scatterplot(x='cv_squared', y='ADI', hue='category', data=adi_cv)

adi_cv.category.value_counts()

