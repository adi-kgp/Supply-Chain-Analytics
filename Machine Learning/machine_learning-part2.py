#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#### Classification
import pandas as pd
import seaborn as sns

banking = pd.read_csv('bank-full.csv')

banking.iloc[0,:]

banking.y.value_counts()

banking.info()

sns.pairplot(banking.iloc[:, [0,5,11,12,13,14,16]], hue='y')

data = banking.iloc[:, [0,5,11,12,13,14]].corr()

sns.heatmap(data)

### Mapping

dict_target = {'yes': 1, 'no':0}

banking['target'] = banking['y'].map(dict_target)

banking.target.value_counts()

banking.isnull().sum().sum()

banking.columns

banking = banking.drop('y', axis=1)

y = banking.target.values

X_ = banking.drop('target', axis=1)

X = pd.get_dummies(X_).values


### without parameter tuning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

model_lr = LogisticRegression()

model_lr.fit(X_train, y_train)

## training score
model_lr.score(X_train, y_train)

##testing score
model_lr.score(X_test, y_test)

### Pre processing
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

## training score
model_lr.score(X_train, y_train)

##testing score
model_lr.score(X_test, y_test)


### Grid Search CV
import numpy as np
from sklearn.model_selection import GridSearchCV

Cs = np.logspace(-5, 5, 20)

grid = {'C': Cs, 'penalty': ['l2']}

model_grid = LogisticRegression()

grid_fit = GridSearchCV(model_grid, grid, cv=6)

grid_fit.fit(X_train, y_train)

grid_fit.best_params_
grid_fit.best_score_

### Area under the curve

## True positive rate = TP / (TP+FN)
## Specificity = TN / (TN+FP)

import scikitplot as skplt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

model = LogisticRegression(C=1.8329807108324339, penalty='l2')

model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(confusion_matrix(y_test, y_predict))

### Frequency
np.unique(y_test, return_counts=True)

tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()

(tn, fp, fn, tp)

y_predicted_probability = model.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_predicted_probability)

y_predicted_probability_both = model.predict_proba(X_test)

skplt.metrics.plot_roc_curve(y_test, y_predicted_probability_both)


## pipelines
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

p_lr = Pipeline([('Imputing',SimpleImputer(missing_values=np.nan, strategy='mean')),
                 ('scaling', StandardScaler()),
                 ('logistic', LogisticRegression())])

p_rf = Pipeline([('Imputing',SimpleImputer(missing_values=np.nan, strategy='mean')),
                 ('scaling', StandardScaler()),
                 ('rf', RandomForestClassifier())])

p_svc = Pipeline([('Imputing',SimpleImputer(missing_values=np.nan, strategy='mean')),
                 ('scaling', StandardScaler()),
                 ('svc', SVC())])

p_KNN = Pipeline([('Imputing',SimpleImputer(missing_values=np.nan, strategy='mean')),
                 ('scaling', StandardScaler()),
                 ('knn', KNeighborsClassifier())])

param_range = range(1,11)

lr_range = np.logspace(-5, 5, 15)

grid_logistic = [{
    'logistic__penalty':['l1', 'l2'],
    'logistic__C' : lr_range,
    'logistic__solver': ['liblinear']
                  }]

grid_rf = [{
        'rf__criterion':['gini', 'entropy'],
        'rf__min__samples': param_range
    }]

grid_svc = [{
        'SVC__kerrnel': ['linear', 'rbf'],
        'SVC__C': param_range
    }]

grid_knn = [{
        'knn__n_neighbors': param_range
    }]

pipes = [p_lr, p_KNN]
grids = [grid_logistic, grid_knn]


fitted_params = []
fitted_score = []
fitted_roc = []

for i in range(0,2):
    model = GridSearchCV(pipes[i], grids[i], cv=3, scoring='accuracy', verbose=10)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    roc = roc_auc_score(y_test, y_pred_prob)
    fitted_params.append(model.best_params_)
    fitted_score.append(model.best_score_)
    fitted_roc.append(roc)

fitted_params
fitted_score
fitted_roc

## random forest and decision trees 
from random import randint

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

param_dict = {
        "max_depth": [3,None],
        "min_samples_leaf": range(1,9),
        "criterion": ["gini", "entropy"]
    }

# without randomized search cv
tree = DecisionTreeClassifier()
rf = RandomForestClassifier()

tree.fit(X_train, y_train)
tree.score(X_train, y_train)
predict_tree = tree.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, predict_tree)

rf.fit(X_train, y_train)
rf.score(X_train, y_train)
predict_rf = rf.predict_proba(X_test)[:,1]
roc_auc_score(y_test, predict_rf)
tree_cv = RandomizedSearchCV(tree, param_dict, cv=5)
rf_cv = RandomizedSearchCV(rf, param_dict, cv=5)

tree_cv.fit(X_train, y_train)
rf_cv.fit(X_train, y_train)

tree_cv.best_score_
rf_cv.best_score_

cv_tree_predict_prob = tree_cv.predict_proba(X_test)[:,1]
roc_auc_score(y_test, cv_tree_predict_prob)

cv_rf_predict_prob = rf_cv.predict_proba(X_test)[:,1]
roc_auc_score(y_test, cv_rf_predict_prob)

## random forest classifier is better than decision tree, randomized search cv offered greater score than normal score, and more time efficient than gridseach cv