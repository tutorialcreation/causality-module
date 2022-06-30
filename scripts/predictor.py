#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from causalnex.structure import DAGClassifier
import joblib

data = load_breast_cancer()


class Predict:
    def __init__(self) -> None:
        pass

    def test_without_causal_nets_lr(self,data):
        clf = LogisticRegression()

        X, y = data.data, data.target
        names = data["feature_names"]
        scores = cross_val_score(clf, X, y, cv=KFold(shuffle=True, random_state=42))
        print(f'MEAN Score: {np.mean(scores).mean():.3f}')

        X = pd.DataFrame(X, columns=names)
        y = pd.Series(y, name="NOT CANCER")
        clf.fit(X, y)
        for i in range(clf.coef_.shape[0]):
            print(f"MEAN EFFECT DIRECTIONAL CLASS {i}:")
            print(pd.Series(clf.coef_[i, :], index=names).sort_values(ascending=False))
        return data


    # In[1]:




    # In[6]:


    def test_with_causal_nets(self,data):
        X, y = data.data, data.target
        names = data["feature_names"]

        clf = DAGClassifier(
            alpha=0.1,
            beta=0.9,
            hidden_layer_units=[5],
            fit_intercept=True,
            standardize=True
        )

        scores = cross_val_score(clf, X, y, cv=KFold(shuffle=True, random_state=42))
        print(f'MEAN Score: {np.mean(scores).mean():.3f}')

        X = pd.DataFrame(X, columns=names)
        y = pd.Series(y, name="NOT CANCER")
        clf.fit(X, y)
        for i in range(clf.coef_.shape[0]):
            print(f"MEAN EFFECT DIRECTIONAL CLASS {i}:")
            print(pd.Series(clf.coef_[i, :], index=names).sort_values(ascending=False))
        clf.plot_dag(True)


    # In[2]:




    # In[8]:


    def save_model(self,clf,filename):
        # save
        joblib.dump(clf, filename) 
        # load
        

    