#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:26:59 2021

@author: matthewyeung
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def percent_change_column(df, column_names):
    for column in column_names:
        df[column] = df[column].pct_change(fill_method='ffill') + 1
    return df

def standardize(df, column_name):
    x = df[column_name].values
    x_scale = StandardScaler().fit_transform(x)
    df_temp = pd.DataFrame(x_scale, columns=column_name, index = df.index)
    df[column_name] = df_temp
    return df

def minmaxstandardized(df, column_name):
    x = df[column_name].values
    x_scale = MinMaxScaler(feature_range=(0.01, 0.99)).fit_transform(x)
    df_temp = pd.DataFrame(x_scale, columns=column_name, index = df.index)
    df[column_name] = df_temp
    return df

def lognormalized(df, column_name):
    df[column_name] = np.log(df[column_name])
    standardize(df, column_name)
    return df

def interpolation(dftest, dftrain, column_name):
    x_train = dftrain[column_name].values
    x_test = dftest[column_name].values
    
    column_min = np.min(x_train, axis=0)
    column_max = np.max(x_train, axis=0)
    column_range = column_max-column_min
    
    df = (x_test - column_min)/(column_range+0.001)
    df_temp = pd.DataFrame(df, columns=column_name)
    dftest[column_name] = df_temp
    return dftest