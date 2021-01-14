#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:37:02 2021

@author: matthewyeung
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# Backtesting, Model in real transactions
def model_strategy(df_buysell, commission, name, training_episode, num_parameter):
    num_year = round(len(df_buysell)/253,1)
    df_buysell['buy_sell'] = 0
    df_buysell = df_buysell.fillna(0)
    
    df = df_buysell
    df['buy'] = df['new_buy'] != 0
    df['pct_change'] = df['price'].pct_change()
    df['percent'] = df['pct_change']*df['buy'] 
    df['cum_perf'] = (df['percent']+1).cumprod()
    SD = df['cum_perf'].std()
    
    for i in range(len(df_buysell)):
        if i == 0:
            if df_buysell['new_buy'].iloc[i] != 0:
                df_buysell['buy_sell'].iloc[i] = -1        # buy = -1
        else:
            if df_buysell['new_buy'].iloc[i-1] != 0 and df_buysell['new_buy'].iloc[i] == 0:
                df_buysell['buy_sell'].iloc[i] = 1         # sell = 1
            elif df_buysell['new_buy'].iloc[i-1] == 0 and df_buysell['new_buy'].iloc[i] != 0:
                df_buysell['buy_sell'].iloc[i] = -1        # buy = -1
    
    #last trade should be close the position
    if df_buysell['buy_sell'].sum() != 0:
        if df_buysell['buy_sell'].iloc[-1] == 0:
            df_buysell['buy_sell'].iloc[-1] = 1  # sell = 1
        elif df_buysell['buy_sell'].iloc[-1] == -1:
            df_buysell['buy_sell'].iloc[-1] = 0  # cancel the trade    
    
    num_trade = len(df_buysell[df_buysell['buy_sell'] == 1])*2
    tran_per_year = num_trade/num_year
    
    under_total_return = (df_buysell['price'].iloc[-1] - df_buysell['price'].iloc[0])/df_buysell['price'].iloc[0]
    under_annual_return =  (under_total_return + 1)**(1/num_year) - 1
    
    df_buysell['com_cost'] = df_buysell['buy_sell'].abs() * df_buysell['price']*(-1)*commission/100
    df_buysell['cost_price'] = df_buysell['com_cost'] + df_buysell['price']*df_buysell['buy_sell']
    
    df_ = df_buysell[df_buysell['cost_price'] != 0].reset_index()
    df_each_trade_pnl = pd.DataFrame(columns = ['Date', 'Long_short', 'PnL'])
    if len(df_) % 2 == 1:
        raise ValueError("not clear the position yet")
    
    record = {'Date':0, 'Long_short':'Long', 'PnL':0}
    df_each_trade_pnl = df_each_trade_pnl.append(record, ignore_index=True)
    for i in range(0,len(df_),2):
        date = df_['index'][i+1]
        longshort = 'Long'
        pnl = (df_['cost_price'][i+1] + df_['cost_price'][i])/np.abs(df_['cost_price'][i])
        record = {'Date':date, 'Long_short':longshort, 'PnL':pnl}
        df_each_trade_pnl = df_each_trade_pnl.append(record, ignore_index=True)
    
    df_each_trade_pnl['ones'] = 1
    df_each_trade_pnl['cum_pnl'] = df_each_trade_pnl['PnL'] + df_each_trade_pnl['ones']
    df_each_trade_pnl['cum_pnl'] = df_each_trade_pnl['cum_pnl'].cumprod()
    
    if num_trade == 0:
        win_ratio = 0
        sharpe_ratio = 0
        max_drawdown = 0
        max_return = 0
        under_total_return = 0
        under_annual_return = 0
        model_total_return = 0
        model_annual_return = 0    
    
    else:
        win = len(df_each_trade_pnl[df_each_trade_pnl['PnL'] > 0])
        loss = len(df_each_trade_pnl[df_each_trade_pnl['PnL'] < 0])
        win_ratio = win/(win+loss+0.001)        
        max_drawdown = df_each_trade_pnl['cum_pnl'].min()
        max_drawdown = (max_drawdown - 1)*100
    
        max_return = df_each_trade_pnl['cum_pnl'].max()
        max_return = (max_return - 1)*100

        model_total_return = df_each_trade_pnl['cum_pnl'].iloc[-1] -1
        model_annual_return = (model_total_return+1)**(1/num_year) - 1
    
        sharpe_ratio = model_total_return/(SD+0.001)

    
    save_list = [name,
                 training_episode,
                 num_parameter,
                 len(df_buysell),
                 num_year,
                 num_trade,
                 tran_per_year,
                 win_ratio*100,
                 max_drawdown,
                 max_return,
                 sharpe_ratio,
                 under_total_return*100,
                 under_annual_return*100,
                 model_total_return*100,
                 model_annual_return*100]
    
    df_save = pd.DataFrame(save_list)
    
    df_save.to_csv(name + '_stats.csv')
    
    
    return df_each_trade_pnl

def cum_perform(cum_price):
    df = pd.DataFrame(cum_price,columns = ['price'])
    df['pct_change'] = df['price'].pct_change()
    df['percent'] = df['pct_change'] + 1
    df['cum_perf'] = df['percent'].cumprod()
    return df

#plot the model return vs Underlying
def return_underlying(df_each_trade_pnl, df_cum_price_val, name):
    plt.figure(figsize=(15,8))
    plt.grid()
    plt.plot(df_each_trade_pnl['Date'], (df_each_trade_pnl['cum_pnl'])*100, label = 'Model')
    plt.plot((df_cum_price_val['cum_perf'])*100, label = 'Underlying')
    plt.title(name + ' - ' + ticker + ' Cumulative Performance')
    plt.xlabel('Time')
    plt.ylabel('Percentage')
    plt.legend()
    plt.savefig(name + '_backtesting_perform.png', dpi = 100)
    plt.show() 
  
#construct a result DataFrame
def construct_corr(cum_rewards_val, 
                   cum_actions_val, 
                   cum_buysell_val, 
                   cum_price_val, 
                   action_0_val, 
                   action_1_val, 
                   action_2_val, 
                   name):
    result_list = [cum_rewards_val,
                   cum_actions_val,
                   cum_buysell_val,
                   cum_price_val,
                   action_0_val,
                   action_1_val,
                   action_2_val,
                   ]
    columns = ['rewards','actions','buysell','Price','actio0','action1','action2']

    for i in range(len(result_list)):
        if i == 0:
            result = pd.DataFrame(result_list[i], columns = [columns[i]])
        else:
            df_ = pd.DataFrame(result_list[i], columns = [columns[i]])
            result = pd.concat([result, df_], axis = 1)

    #Adding the training data
    result.to_csv(name + '_raw_data.csv')
    df_concat = df_train[-200 + timestep: -1].reset_index(drop=True)
    result = pd.concat([result, df_concat], axis = 1)
    result.drop(['price'], axis=1, inplace=True)
    result_corr = result.corr()
    result_corr.to_csv(name+'_corr.csv')
