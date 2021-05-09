# -*- coding: utf-8 -*-
"""
Created on Sun May  9 19:31:33 2021

@author: saniy
"""

import pandas as pd
heart = pd.read_csv('heart.csv')

df = heart.copy()
target = 'target'

# Separating X and y
X = df.drop('target', axis=1)
Y = df['target']

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

import pickle
pickle.dump(clf, open('heart_clf.pkl', 'wb'))