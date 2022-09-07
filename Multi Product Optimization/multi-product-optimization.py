#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import inventorize as inv
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

multi = pd.read_csv('multvariate_slides.csv')

multi.describe()

multi.iloc[:,1:5].describe()

X = multi.iloc[:, 1:5]
X = sm.add_constant(X)

multi.columns

model_product1 = sm.OLS(multi[['sales_product1']], X).fit()
model_product1.summary()

model_product2 = sm.OLS(multi[['sales_product2']], X).fit()
model_product2.summary()

model_product3 = sm.OLS(multi[['sales_product3']], X).fit()
model_product3.summary()

model_product4 = sm.OLS(multi[['sales_product4']], X).fit()
model_product4.summary()

#### Multinomial logit models

multi_choice = pd.read_csv('multi_slides.csv')

multi_choice

choices = inv.Multi_Competing_optimization(X = multi_choice.iloc[:1000, 0:4], 
                                 y = multi_choice.loc[:1000, 'choice'], 
                                 n_variables = 4, 
                                 initial_products_cost = [40, 60, 70, 100])

choices_without_cost = inv.Multi_Competing_optimization(X = multi_choice.iloc[:1000, 0:4], 
                                 y = multi_choice.loc[:1000, 'choice'], 
                                 n_variables = 4, 
                                 initial_products_cost = [0,0,0,0])











