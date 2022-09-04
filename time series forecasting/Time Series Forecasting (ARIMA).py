#!/usr/bin/env python
# coding: utf-8

# ## Time Series Forecasting models (ARIMA)
# 
# ARIMA and Exponential Smoothing are the two most widely used approaches to 
#time series forecasting, and provide complementary approaches to the problem.
# While exponential smoothing models are based on a description of the trend 
# and seasonality in the data, ARIMA models aim to describe the autocorrelations 
# in the data. In this script, we study the ARIMA forecasting technique.


import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm # has different time series models 
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

time_series = pd.read_csv("timeseries_rev.csv", parse_dates = True)
time_series.head()

# set date column datatype
time_series['date'] = pd.to_datetime(time_series['date'])
## set the date as index
time_series = time_series.set_index('date')


monthly_series = time_series.total_revenue.resample('M').sum()
monthly_series.head()


plt.figure(figsize = (6,2))
monthly_series.plot();


# components of time series (trend, seasonal and residual) , using the below methods from the statsmodels.api library
components = sm.tsa.seasonal_decompose(monthly_series)
plt.rcParams["figure.figsize"] = (10,6)
components.plot();

trend = components.trend
trend.head()

seasonality = components.seasonal
seasonality.head()


remainder = components.resid
remainder.head()


# Checking for seasonality
plt.figure(figsize=(8,3))
monthly_series.plot()
monthly_series.rolling(window=12).mean().plot()
monthly_series.rolling(window=12).std().plot();


# We can argue that the mean is stationary by looking at the graph above , also standard deviation seemd to change very slightly. But we can confirm this result using AD Fuller Test.

ad_fuller_test = sm.tsa.stattools.adfuller(monthly_series, autolag="AIC")
ad_fuller_test # null hypothesis: data is not stationary, mean is varying
#alternate hypothesis: data is stationary, mean is changing 
#test: pvalue is less than 0.05, then we reject null hypothesis (that is accept alternate hypothesis) 


# The second value in the ad fuller test above is the pvalue, which is less than 0.05 meaning that we reject null hypothesis and come to the conlusion that our data (monthly_series) is stationary, which was how the graph was looking above.

## ACF and PACF plots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(monthly_series);

# plot_pacf(monthly_series) some technical error related to dataset, need to fix this 


## moving averages model
model_ma = sm.tsa.statespace.SARIMAX(monthly_series, order=(0,0,1)) 
results = model_ma.fit() 
results.aic


### Autoregressive model with order 1
model_ar = sm.tsa.statespace.SARIMAX(monthly_series, order=(1,0,0))
results_ar1 = model_ar.fit() 
results_ar1.aic


### Autoregressive model with order 2
model_ar = sm.tsa.statespace.SARIMAX(monthly_series, order=(2,0,0))
results_ar2 = model_ar.fit() 
results_ar2.aic


## ARMA model 
model_arma = sm.tsa.statespace.SARIMAX(monthly_series, order=(1,0,1))
results_arma = model_arma.fit() 
results_arma.aic


### ARIMA model
model_arima = sm.tsa.statespace.SARIMAX(monthly_series, order=(1,1,1))
results_arima = model_arima.fit() 
results_arima.aic


# ### ARIMA Diagnostics

# The best model we have is the ARIMA model.

results_arima.plot_diagnostics(figsize=(15, 12));


# ### ARIMA Grid Search 

P=D=Q=p=d=q= range(0,3)
S = 12
combinations = list(itertools.product(P,D,Q,p,d,q))

arima_order = [(x[0], x[1], x[2]) for x in combinations]


seasonal_order = [(x[3], x[4], x[5], S) for x in combinations]


results_data = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D', 'Q', 'AIC'])
## length of combinations
len(combinations)


for i in range(len(combinations)):
    try:
        
        model = sm.tsa.statespace.SARIMAX(monthly_series, order=arima_order[i],
                                     seasonal_order = seasonal_order[i])
        result = model.fit()
        results_data.loc[i, 'p'] = arima_order[i][0]
        results_data.loc[i, 'd'] = arima_order[i][1]
        results_data.loc[i, 'q'] = arima_order[i][2]
        results_data.loc[i, 'P'] = seasonal_order[i][0]
        results_data.loc[i, 'D'] = seasonal_order[i][1]
        results_data.loc[i, 'Q'] = seasonal_order[i][2]
        results_data.loc[i, 'AIC'] = result.aic
    except:
        continue


results_data



results_data[results_data['AIC']== min(results_data.AIC)]


# The model with minimum AIC is the model with no autoregressive and moving average coefficients, but with non-zero differencing coefficients and seasonality 


# Lets try the best models 
best_model = sm.tsa.statespace.SARIMAX(monthly_series,
                                      order=(0,1,0),
                                        seasonal_order=(0,2,0,12))


results = best_model.fit()


fitting = results.get_prediction(start='2009-12-31')
fitting_mean = fitting.predicted_mean
fitting_mean


forecast = results.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean

fitting_mean.plot(figsize=(8,3), label='Fitting');
monthly_series.plot(label='Actual');
forecast_mean.plot(label='Forecast');
plt.legend(loc='upper left');


mean_absolute_error = abs(monthly_series - fitting_mean).mean()
mean_absolute_error


## Testing the ARIMA model error
model_arima = sm.tsa.statespace.SARIMAX(monthly_series, order=(1,1,1))
results_arima = model_arima.fit() 
results_arima.aic # aic is the balance between model complexity and the errors, so its not necessary to have least aic and also have least error

fitting = results_arima.get_prediction(start='2009-12-31')
fitting_mean = fitting.predicted_mean
mae_arima = abs(monthly_series - fitting_mean).mean()
mae_arima


# The Mean Absolute Error of ARIMA model is **LESS** than the model with least AIC (that we got from the grid search), meaning there is no correlation between AIC and MAE, that is to say the model with the least AIC does not mean it will produce least MAE.





