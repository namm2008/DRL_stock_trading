#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:22:57 2021

@author: matthewyeung
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as web

#RSI
def rsi(dataset, window_length):
    # Get the difference in price from previous step
    delta = dataset.diff()
    # Get rid of the first row, which is NaN since it did not have a previous 
    # row to calculate the differences
    delta = delta[1:] 

    # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    # Calculate the EWMA
    roll_up = up.ewm(span=window_length).mean()
    roll_down = down.abs().ewm(span=window_length).mean()

    # Calculate the RSI based on EWMA
    RS = roll_up / roll_down
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def find_na(df):
    col_name = df.columns
    for col in col_name:
        nan_num = df[col].isnull().sum()
        print('Column Name: {}, NaN Number: {}'.format(col, nan_num))

def find_zero(df):
    col_name = df.columns
    for col in col_name:
        zero_num = len(df[df[col]==0])
        print('Column Name: {}, Zero Number: {}'.format(col, zero_num)) 

#Download data from the web
def stock_dataset_dl(ticker, start, end, ema_short, ema_long, day_data= True):
    #import data with pandas dataframe
    df = web.DataReader(ticker, 'yahoo', start, end)
    df['price'] = df['Close']
    df['ema_st'] = df['Close'].ewm(span=ema_short, adjust=False).mean()
    df['ema_lg'] = df['Close'].ewm(span=ema_long, adjust=False).mean()
    df['MACD'] = df['ema_st'] - df['ema_lg']
    df['rsi'] = rsi(df['Close'],20)
    df['Percent_chg'] = df['Close'].pct_change()
    
    df['high_low'] = df['High'] - df['Low']
    
    #handle 0 high_low value
    df['high_low_1'] = df['high_low']
    for i in range(len(df)):
        if df['high_low_1'].iloc[i] == 0:
            df['high_low'].iloc[i] = 0.1
        
    #handling time
    df = df.reset_index()
    if day_data == True:
        df['timestp'] = pd.to_datetime(df['Date'])
        df['daytime'] = df['timestp'].dt.dayofweek
        df['day'] = df['timestp'].dt.day
        #df['hour'] = df['timestp'].dt.hour
        #df['minute'] = df['timestp'].dt.minute
    
        df_train = df[['price', 'Close', 'high_low', 'Volume', 'ema_st','ema_lg',
                   'MACD','rsi','day','daytime','Percent_chg']]
    else: 
        df_train = df[['price', 'Close', 'high_low', 'Volume', 'ema_st','ema_lg','MACD','rsi','Percent_chg']]
    
    #fill the NaN with 0
    df_train = df_train.fillna(0)
    
    return df_train
