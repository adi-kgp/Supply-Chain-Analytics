# -*- coding: utf-8 -*-
## Single Exponential Smoothing

""" The simplest of the exponential smoothing methods is naturally called Simple
Exponential Smoothing. This method is suitable for forecasting data with no clear
trend or seasonal pattern
The other types of Exponential Smoothing methods include  Holt's Linear Trend 
Method, Holt-Winter's Additive method 
"""

import statsmodels as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

time_series = pd.read_csv("timeseries_rev.csv", parse_dates = True)
time_series.head()

# set date column datatype
time_series['date'] = pd.to_datetime(time_series['date'])
## set the date as index
time_series = time_series.set_index('date')

monthly_series = time_series.total_revenue.resample('M').sum()

model_expo1 = sm.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend='add',
                                                      seasonal='add',seasonal_periods=12)

model_expo2 = sm.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend='mul',
                                                      seasonal='add',seasonal_periods=12)

model_expo3 = sm.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend='add',
                                                      seasonal='mul',seasonal_periods=12)

model_expo4 = sm.tsa.holtwinters.ExponentialSmoothing(monthly_series, trend='mul',
                                                      seasonal='mul',seasonal_periods=12)

results_1 = model_expo1.fit()
results_2 = model_expo2.fit()
results_3 = model_expo3.fit()
results_4 = model_expo4.fit()

results_1.summary()
results_2.summary()
results_3.summary()
results_4.summary()

fit1 = model_expo1.fit().predict(0, len(monthly_series))
fit2 = model_expo2.fit().predict(0, len(monthly_series))
fit3 = model_expo3.fit().predict(0, len(monthly_series))
fit4 = model_expo4.fit().predict(0, len(monthly_series))

# Let's get the MAE for every model fit
mae1 = abs(monthly_series - fit1).mean()
mae2 = abs(monthly_series - fit2).mean()
mae3 = abs(monthly_series - fit3).mean()
mae4 = abs(monthly_series - fit4).mean()

forecast=model_expo1.fit().predict(0, len(monthly_series) + 12)

monthly_series.plot(label="Actual")
forecast.plot(label="Forecast")
plt.legend(loc='upper left')

""" To conclude, we can see that the Exponential Smoothing model is more accurately
depicting the data than the ARIMA model for this dataset. """ 




















