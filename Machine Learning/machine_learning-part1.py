#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rfm = pd.read_csv('rfm_revised.csv')

rfm.columns

X = rfm[['frequency', 'monetary', 'recency']]

km = KMeans(n_clusters=3, n_init=10, max_iter = 300, tol=0.0001)

fitting = km.fit_predict(X)

X['centroids'] = fitting

sns.pairplot(data=X, hue='centroids')

sse = []
X = rfm[['frequency', 'monetary', 'recency']]

for k in range(1,11):
    kmeans = KMeans(n_clusters=k, n_init=10, max_iter = 300, tol=0.0001)
    a = kmeans.fit(X)
    sse.append(a.inertia_)
    
sse

plt.plot(range(1,11), sse)

#### Regression

retail_clean = pd.read_csv('retail_clean.csv')

retail_clean['InvoiceDate'] = pd.to_datetime(retail_clean['InvoiceDate'])

retail_clean['date'] = retail_clean['InvoiceDate'].dt.strftime('%Y-%m-%d')

retail_clean['date'] = pd.to_datetime(retail_clean['date'])

daily_revenue = retail_clean.groupby(['date']).agg(total_revenue = ('Revenue', 'sum')).reset_index()

daily_revenue['month'] = daily_revenue['date'].dt.month
daily_revenue['dayofweek'] = daily_revenue['date'].dt.dayofweek
daily_revenue['trend'] = range(1, daily_revenue.shape[0]+1)

daily_revenue['month'] = daily_revenue['month'].astype('category')

weekdays = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 
            5: 'Saturday', 6: 'Sunday'}

daily_revenue['dayofweek1'] = daily_revenue['dayofweek'].map(weekdays)

daily_revenue = daily_revenue.drop('dayofweek', axis=1)

daily_Revenue_encoded = pd.get_dummies(daily_revenue)

daily_Revenue_encoded.columns

plt.plot(daily_revenue['date'], daily_revenue['total_revenue'])

from sklearn.model_selection import train_test_split

X = daily_Revenue_encoded.drop(['date','total_revenue'], axis=1).values
y = daily_Revenue_encoded['total_revenue'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)

len(X_train), len(X_test)


### Model Implementation

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

model_linear = LinearRegression().fit(X_train, y_train)
model_lasso = Lasso(alpha=0.006, normalize=True, tol=0.000001, max_iter=1000).fit(X_train, y_train)
model_tree = DecisionTreeRegressor().fit(X_train, y_train)
model_knn = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)

## Training Score
model_linear.score(X_train, y_train)
model_lasso.score(X_train, y_train)
model_tree.score(X_train, y_train)
model_knn.score(X_train, y_train)

## Testing Score
model_linear.score(X_test, y_test)
model_lasso.score(X_test, y_test)
model_tree.score(X_test, y_test)
model_knn.score(X_test, y_test)

## Predictions
y_linear = model_linear.predict(X_test)
y_lasso = model_lasso.predict(X_test)
y_tree = model_tree.predict(X_test)
y_knn = model_knn.predict(X_test)

## mean squared error
mean_squared_error(y_test, y_linear)
mean_squared_error(y_test, y_lasso)
mean_squared_error(y_test, y_tree)
mean_squared_error(y_test, y_knn)

# mean absolute error
mean_absolute_error(y_test, y_linear)
mean_absolute_error(y_test, y_lasso)
mean_absolute_error(y_test, y_tree)
mean_absolute_error(y_test, y_knn)

## parameter tuning for KNN

MAE_training = []
MAE_testing = []
neighbors = range(1,20)

for n in neighbors:
    model = KNeighborsRegressor(n_neighbors=n).fit(X_train, y_train)
    y_predict_training = model.predict(X_train)
    y_predict_testing = model.predict(X_test)
    training = mean_absolute_error(y_predict_training, y_train)
    testing = mean_absolute_error(y_predict_testing, y_test)
    MAE_training.append(training)
    MAE_testing.append(testing)
    
plt.plot(neighbors, MAE_training, label='training')
plt.plot(neighbors, MAE_testing, label='testing')
plt.legend(loc='lower right')

## Parameter tuning for Lasso
import numpy as np

alphas = np.linspace(0, 1, 100)

MAE_training = []
MAE_testing = []
model_scores = []

for alpha in alphas:
    model = Lasso(alpha=alpha, normalize=True).fit(X_train, y_train)
    y_predict_training = model.predict(X_train)
    y_predict_testing = model.predict(X_test)
    scores = model.score(X_train, y_train)
    training = mean_absolute_error(y_predict_training, y_train)
    testing = mean_absolute_error(y_predict_testing, y_test)
    MAE_training.append(training)
    MAE_testing.append(testing)
    model_scores.append(scores)
    
plt.plot(alphas, MAE_training, label='training')
plt.plot(alphas, MAE_testing, label='testing')
plt.legend(loc='lower right')
    
alpha_data = pd.DataFrame({'alpha': alphas, 'training': MAE_training, 
                           'testing': MAE_testing, 'scores': model_scores})

alpha_data[alpha_data.scores==max(alpha_data.scores)]

model_alpha = Lasso(alpha=0.8).fit(X_train, y_train)

model_alpha.coef_

names = daily_Revenue_encoded.drop(['date', 'total_revenue'], axis=1).columns

plt.plot(names, model_alpha.coef_)
plt.xticks(rotation=90)

data_coef = pd.DataFrame({'names': names, 'coef': model_alpha.coef_})





