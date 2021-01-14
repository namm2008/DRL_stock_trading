#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:32:35 2021

@author: matthewyeung
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

#plot graph
#learning curve
def learning_curve(avg_losses, name):
    plt.figure(figsize = (15,8))
    plt.plot(avg_losses, color='g' ,label='Loss')
    plt.grid()
    plt.title(name + ' - Training Loss')
    plt.legend()
    plt.savefig(name + '_Training_Loss.png', dpi = 100)
    plt.show()

#testing Loss
def testing_loss(avg_losses_val, name):
    plt.figure(figsize = (15,8))
    plt.plot(avg_losses_val, color='y' ,label='Loss')
    plt.grid()
    plt.title(name + ' - Testing Loss')
    plt.legend()
    plt.savefig(name + '_Testing_Loss.png', dpi = 100)
    plt.show()

def clone_detach(cum_list):
    cum_list_1 = []
    for i in cum_list:
        cum_list_1.append(i.cpu().clone().detach().numpy())
    return cum_list_1

#rewards graph
def reward_time(cum_rewards_val, name):
    plt.figure(figsize = (15,8))
    plt.plot(cum_rewards_val, color='b' ,label='Rewards')
    plt.grid()
    plt.title(name + ' - Rewards / Time')
    plt.legend()
    plt.savefig(name + '_rewards.png', dpi = 100)
    plt.show()

def reward_price(cum_rewards_val, cum_price_val, name):
    # reward price graph
    fig,ax = plt.subplots(figsize = (15,8))
    plt.grid()
    # make a plot
    ax.plot(cum_rewards_val, color="blue")
    # set x-axis label
    ax.set_xlabel("Episode",fontsize=14)
    # set y-axis label
    ax.set_ylabel("Rewards",color="blue",fontsize=14)
    # set title
    ax.set_title(name + ' - ' + 'Rewards vs Price')
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(cum_price_val , color="orange")
    ax2.set_ylabel("Price",color="orange",fontsize=14)

    plt.savefig(name + '_rewards_price.png', dpi = 100)
    plt.show()

#actions price graph
def action_price(action_0_val, action_1_val, action_2_val,cum_price_val, name):
    fig,ax = plt.subplots(figsize = (15,8))
    plt.grid()
    # make a plot
    ax.plot(action_0_val, color="gray", label = 'Action 0')
    ax.plot(action_1_val, color="green", label = 'Action 1')
    ax.plot(action_2_val, color="red", label = 'Action 2')
    #ax.plot(cum_buysell_test, color='b' ,marker = 'x', label='Buy/sell')
    #ax.plot(vol, color="purple", label = 'Volume')
    plt.legend(loc='upper right')
    # set x-axis label
    ax.set_xlabel("Time",fontsize=14)
    # set y-axis label
    ax.set_ylabel("Q values",color="black",fontsize=14)
    # set title
    ax.set_title(name + ' - Testing - Action Q value vs Price',fontsize=14)
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(cum_price_val , color="orange", label = 'Price')
    ax2.set_ylabel("Price",color="orange",fontsize=14)
    plt.legend(loc='upper left')

    plt.savefig(name + '_testing_qvalue_price.png', dpi = 100)
    plt.show()

def buysell_price(price, buysell):
    buysell = pd.DataFrame(buysell, columns = ['Buy_sell'])
    price = pd.DataFrame(price, columns = ['price'])
    df = pd.concat([price , buysell], axis =1)
    df['new_buy'] = df['Buy_sell']*df['price']
    for i in range(len(df)):
        if df['Buy_sell'].iloc[i] == 0:
            df['new_buy'].iloc[i] = None
    df = df[['price','new_buy']]
    return df

#Buy and sell with price
def buy_price(buysell_val, name):
    plt.figure(figsize = (15,8))
    plt.grid()
    plt.plot(buysell_val['price'], color='b' ,label='Price')
    plt.plot(buysell_val['new_buy'], color='g' ,label='Buy', marker = 'x')
    plt.title(name + ' - Buy and Sell action with Price',fontsize = 14)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(name + '_testing_action_price.png', dpi = 100)
    plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)