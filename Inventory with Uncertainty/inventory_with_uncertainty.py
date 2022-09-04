#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import inventorize3 as inv
import datetime as dt
import numpy as np

retail = pd.read_csv('online_retail2.csv')

retail = retail.drop_duplicates()

retail = retail.dropna(axis=0)

retail.info()

retail.InvoiceDate

retail.InvoiceDate = pd.to_datetime(retail.InvoiceDate)

retail['date'] = retail.InvoiceDate.dt.strftime('%Y-%m-%d')

retail.date = pd.to_datetime(retail.date)

max_date = max(retail.date)

last_four = retail[retail['date'] > '2011-08-01']


# Average demand 
a = retail.groupby(['date', 'Description']).agg(total_daily = ('Quantity', np.sum)).reset_index()

grouped = a.groupby(['Description']).agg(average=('total_daily', np.mean),
                                         sd = ('total_daily', 'std')).reset_index()


 

