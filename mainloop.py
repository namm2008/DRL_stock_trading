#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: matthewyeung
"""

if __name__ == "__main__": 
    
    import math
    import random
    import numpy as np
    import pandas as pd
    import pandas_datareader.data as web
    import matplotlib.pyplot as plt
    from collections import namedtuple
    from itertools import count
    from PIL import Image
    from IPython.display import clear_output
    from time import sleep
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import torchvision.transforms as T
    from torch.utils.data import Dataset, DataLoader
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    
    #from action_selection import select_action, select_action_val, reward_action_train
    #from action_selection import *
    from attention_gru import *
    from attention_lstm import *
    from Backtest import *
    #from Optimization import *
    from replay_memory import *
    from trading_environment import *
    from transformation import *
    from Utility_function import *
    import config
    from config import hyperpara
    
    
    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #Initialize all the hyperparameters
    BATCH_SIZE = 64
    learning_rate = 0.0001
    GAMMA = 0.9
    EPS_START = 0.99
    EPS_END = 0.01
    EPS_DECAY = 1000
    cost_percent = 0.25
    capacity = 50000
    capacity_size = 1000
    alpha = 0.7
    beta_start = 0.4
    beta_frames = 10000
    
    output_size = 3
    timestep = 10
    n_layers = 2
    encoder_hidden_dim = 25
    decoder_hidden_dim = 13
    output_size = 3
    bidirectional = True
    
    '''
    model_list = ['RL_LSTM_Attention', 'RL_GRU_Attention']
    PER_list = ['noPER' , 'PER']
    data_feature_list = ['increasing', 'decreasing', 'normal']
    data_time_data_list = ['DT', 'noDT']
    '''
    # define the training and testing model, all the training and test result will be saved
    model_list = ['RL_GRU_Attention']
    PER_list = ['PER']
    data_feature_list = ['normal']
    data_time_data_list = ['noDT']
    
    #Mainloop
    #For loop to loop all the models for GRU attention and LSTM attention models
    for model in model_list:
        for PER in PER_list:
            for data_feature in data_feature_list:
                for data_time_data in data_time_data_list:
                    name = model + '_' + PER +'_' + data_feature + '_' + data_time_data
                    
                    print(name)
                    if data_feature == 'increasing':
                        start = '2006-01-01'
                        end = '2018-08-30'
                        ticker = 'AMZN'
                    elif data_feature == 'decreasing':
                        start = '1998-01-01'
                        end = '2019-12-31'
                        ticker = 'F'
                    elif data_feature == 'normal':
                        start = '2005-01-01'
                        end = '2019-12-31'
                        ticker = 'GM'                    
                    ema_short = 9
                    ema_long = 40
                    
                    if data_time_data == 'DT':
                        data_time = True
                        col_list_minmax = ['Close','Volume','high_low', 'ema_st','ema_lg', 'rsi', 'day', 'daytime']
                    elif data_time_data == 'noDT':
                        data_time = False
                        col_list_minmax = ['Close','Volume','high_low', 'ema_st','ema_lg', 'rsi']
                    df = stock_dataset_dl(ticker, start, end, ema_short, ema_long, data_time)
                    
                    #Train-Test Split
                    train_test_ratio = 0.1
                    dataset_len = len(df)
                    test_length = round(train_test_ratio*dataset_len)
                    df_test = df.iloc[-test_length:,:].reset_index(drop=True)
                    df_train = df.iloc[:(-test_length),:]
                    
                    # minmaxstandardize
                    df_test = interpolation(df_test, df_train, col_list_minmax)
                    df_train = minmaxstandardized(df_train, col_list_minmax)
                    
                    # delete row 0 to 4 for rsi=0
                    df_train.drop([0,1,2,3],inplace = True)
    
                    #reset index
                    df_train.reset_index(drop=True, inplace=True)
                    df_test.reset_index(drop=True, inplace=True)
                    
                    #define dataset and testset
                    dataset = torch.tensor(df_train.iloc[:,1:].values, dtype=torch.float32, device=device)
                    testset = torch.tensor(df_test.iloc[:,1:].values, dtype=torch.float32, device=device)
                    
                    #Initialize the initial features
                    feature_number = torch.tensor((dataset.shape[1]), device = device, dtype=torch.int32)
                    
                    #Inilialize the policy Net and Target Net
                    if model == 'RL_LSTM_Attention':
                        policy_net = LSTM_Attention(feature_number, n_layers, timestep, 
                                                    encoder_hidden_dim, decoder_hidden_dim, output_size, 
                                                    bidirectional = True, drop_prob=0.5).to(device)
                        target_net = LSTM_Attention(feature_number, n_layers, timestep, 
                                                   encoder_hidden_dim, decoder_hidden_dim, output_size, 
                                                   bidirectional = True, drop_prob=0.5).to(device)
    
                    elif model == 'RL_GRU_Attention':
                        policy_net = GRU_Attention(feature_number, n_layers, timestep, 
                                                    encoder_hidden_dim, decoder_hidden_dim, output_size, 
                                                    bidirectional = True, drop_prob=0.5).to(device)
                        target_net = GRU_Attention(feature_number, n_layers, timestep, 
                                                   encoder_hidden_dim, decoder_hidden_dim, output_size, 
                                                   bidirectional = True, drop_prob=0.5).to(device) 
                    target_net.load_state_dict(policy_net.state_dict())
                    target_net.eval()
                    
                    #Define the Optimizer and initialize the replay memory
                    optimizer = optim.RMSprop(policy_net.parameters(),lr=learning_rate)
                    if PER == 'PER':
                        memory = PrioritizedReplayMemory(capacity, alpha, beta_start, beta_frames)
                    elif PER == 'noPER':
                        memory = ReplayMemory(capacity_size)
                    
                    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'longshort'))
                    
                    para = hyperpara(memory, policy_net, target_net, Transition, timestep, optimizer)
                    
                    # Count parameters:
                    num_parameter = count_parameters(policy_net)
                    
                    #Training Loop:
                    iterate_from = 0
                    iterate_end = len(dataset)-timestep -1
                    num_episodes = 20
                    avg_losses = []
                    epi_losses = []
                    for i_episode in range(num_episodes):
                        # Initialize the environment and state
                        state = state_loader(dataset, timestep, iterate_from)
                        #hidden = hidden_1 = policy_net.init_hidden(BATCH_SIZE)
                        losses = []
                        avg_loss = 0
                        losss = 0
                        epi_loss = 0
                        en_hidden_test = policy_net.init_hidden_encoder(1)
                        de_hidden_test = policy_net.init_hidden_decoder(1)
        
                        config.steps_done = 0
                        config.EPS_START, config.EPS_END, config.EPS_DECAY = EPS_START, EPS_END, EPS_DECAY
                        for t in range(iterate_from, iterate_end):
                            # Select and perform an action
                            a = para.select_action(state, en_hidden_test, de_hidden_test)
                            en_hidden = en_hidden_1 = policy_net.init_hidden_encoder(BATCH_SIZE)
                            de_hidden = de_hidden_1 = policy_net.init_hidden_decoder(BATCH_SIZE)
                            config.en_hidden, config.en_hidden_1, config.de_hidden, config.de_hidden_1, config.losss = en_hidden, en_hidden_1, de_hidden, de_hidden_1, losss
                            
                            # Observe new state
                            next_state = state_loader(dataset, timestep, t + 1)
                            
                            # handle the long-short position and add the corresponding position to the replay memory
                            if len(para.memory) == 0 or para.memory.last_ls() == 0:
                                if a == 0:
                                    reward = para.reward_action_train(df_train['price'], a, t, cost_percent)
                                    ls = torch.tensor(0,device=device, dtype=torch.int64)
                                    para.memory.push(Transition(state, a.view(1,1), next_state, reward.view(1,1), ls))
                                elif a == 1:
                                    reward = para.reward_action_train(df_train['price'], a, t, cost_percent)
                                    ls = torch.tensor(1,device=device, dtype=torch.int64)
                                    memory.push(Transition(state, a.view(1,1), next_state, reward.view(1,1), ls))
                    
                            elif para.memory.last_ls() == 1:
                                if a == 0:
                                    reward = para.reward_action_train(df_train['price'], a, t, cost_percent)
                                    ls = torch.tensor(0,device=device, dtype=torch.int64)
                                    para.memory.push(Transition(state, a.view(1,1), next_state, reward.view(1,1), ls))
                                elif a == 2:
                                    reward = para.reward_action_train(df_train['price'], a, t, cost_percent)
                                    ls = torch.tensor(1,device=device, dtype=torch.int64)
                                    para.memory.push(Transition(state, a.view(1,1), next_state, reward.view(1,1), ls))
                            
                            # Move to the next state
                            state = next_state
                            
                            # Perform optimization (on the target network)
                            if PER == 'PER':
                                para.optimize_model_per( GAMMA, BATCH_SIZE)
                            elif PER == 'noPER':
                                para.optimize_model_noper( GAMMA, BATCH_SIZE)
                            losses.append(config.losss)
                            epi_loss += config.losss
                            
                            # Update the target network, copying all weights and biases in DQN
                            if t % 20 == 0:
                                para.target_net.load_state_dict(para.policy_net.state_dict())
                                #Save Model
                                save_name = name
                                torch.save({'policy_net': para.policy_net.state_dict(),
                                            'target_net': para.target_net.state_dict(),
                                            'optimizer': para.optimizer.state_dict()}, save_name + '.pt')
                                avg_loss = np.average(losses)
                                avg_losses.append(avg_loss)
                                losses = []
                        
                        epi_losses.append(epi_loss)
                        if epi_loss == np.min(epi_losses):
                            save_name = name + '_best'
                            torch.save({'policy_net': para.policy_net.state_dict(),
                                        'target_net': para.target_net.state_dict(),
                                        'optimizer': para.optimizer.state_dict()}, save_name + '.pt')
                        if i_episode > 3:
                            if np.min(epi_losses) == epi_losses[-3]:
                                break
                    training_episode = len(epi_losses)
                    
                    np.savetxt(name + "_train_loss.csv", np.column_stack((avg_losses)), delimiter=",", fmt='%s')
                    
                    print('finish training')
                    
                    #save learning curve
                    learning_curve(avg_losses, name)
                    
                    #Testing the model
                    iterate_from = 0
                    iterate_end = len(testset)-timestep - 1
                    num_episodes = 1
    
                    avg_losses_val = []
                    cum_rewards_val=[]
                    cum_actions_val = []
                    cum_buysell_val = []
                    cum_price_val = []
                    action_0_val = []
                    action_1_val = []
                    action_2_val = []
                    for i_episode in range(num_episodes):
                        # Initialize the environment and state
                        state = state_loader(testset, timestep, iterate_from)
                        #hidden = hidden_1 = policy_net.init_hidden(BATCH_SIZE)
                        losses = []
                        avg_loss = 0
                        losss = 0
                        en_hidden_test = policy_net.init_hidden_encoder(1)
                        de_hidden_test = policy_net.init_hidden_decoder(1)
                        rewards = torch.tensor(0, device = device, dtype = torch.float32)
                        config.en_hidden_test, config.de_hidden_test = en_hidden_test, de_hidden_test
    
                        for t in range(iterate_from, iterate_end):
    
                            # Select and perform an action
                            a,a0,a1,a2 = para.select_action_val(state, en_hidden_test, de_hidden_test)
                            en_hidden = en_hidden_1 = policy_net.init_hidden_encoder(BATCH_SIZE)
                            de_hidden = de_hidden_1 = policy_net.init_hidden_decoder(BATCH_SIZE)
                            config.en_hidden, config.en_hidden_1, config.de_hidden, config.de_hidden_1, config.losss = en_hidden, en_hidden_1, de_hidden, de_hidden_1, losss
            
                            # Observe new state
                            next_state = state_loader(testset, timestep, t + 1)
            
                            # handle the long-short position and add the corresponding position to the replay memory
                            if len(para.memory) == 0 or para.memory.last_ls() == 0:
                                if a == 0:
                                    reward = para.reward_action_train(df_test['price'], a, t, cost_percent)
                                    ls = torch.tensor(0,device=device, dtype=torch.int64)
                                    para.memory.push(Transition(state, a.view(1,1), next_state, reward.view(1,1), ls))
                                elif a == 1:
                                    reward = para.reward_action_train(df_test['price'], a, t, cost_percent)
                                    ls = torch.tensor(1,device=device, dtype=torch.int64)
                                    para.memory.push(Transition(state, a.view(1,1), next_state, reward.view(1,1), ls))
                    
                            elif memory.last_ls() == 1:
                                if a == 0:
                                    reward = para.reward_action_train(df_test['price'], a, t, cost_percent)
                                    ls = torch.tensor(0,device=device, dtype=torch.int64)
                                    para.memory.push(Transition(state, a.view(1,1), next_state, reward.view(1,1), ls))
                                elif a == 2:
                                    reward = para.reward_action_train(df_test['price'], a, t, cost_percent)
                                    ls = torch.tensor(1,device=device, dtype=torch.int64)
                                    para.memory.push(Transition(state, a.view(1,1), next_state, reward.view(1,1), ls))
            
                            rewards += reward
            
    
                            # Move to the next state
                            state = next_state
    
                            # Perform optimization (on the target network)
                            if PER == 'PER':
                                para.optimize_model_per(GAMMA, BATCH_SIZE)
                            elif PER == 'noPER':
                                para.optimize_model_noper(GAMMA, BATCH_SIZE)
                            losses.append(config.losss)
            
                            # Update the target network, copying all weights and biases in DQN
                            if t % 10 == 0:
                                para.target_net.load_state_dict(para.policy_net.state_dict())
                
                                #Save Model
                                save_name = name + '_best'
                                torch.save({'policy_net': para.policy_net.state_dict(),
                                            'target_net': para.target_net.state_dict(),
                                            'optimizer': para.optimizer.state_dict()}, save_name + '.pt')
                                avg_loss = np.average(losses)
                                avg_losses_val.append(avg_loss)
    
                                losses = []
            
            
                            current_price = df_test['price'][t + timestep]
    
                            cum_rewards_val.append(rewards.clone().detach())
                            cum_actions_val.append(a[0])
                            cum_buysell_val.append(ls)
                            cum_price_val.append(current_price)
                            action_0_val.append(a0)
                            action_1_val.append(a1)
                            action_2_val.append(a2)                
                    
                    
                    #save testing loss
                    testing_loss(avg_losses_val, name)
                    
                    #clone to cpu
                    cum_rewards_val = clone_detach(cum_rewards_val)
                    cum_actions_val = clone_detach(cum_actions_val)
                    cum_buysell_val = clone_detach(cum_buysell_val)
                    action_0_val = clone_detach(action_0_val)
                    action_1_val = clone_detach(action_1_val)
                    action_2_val = clone_detach(action_2_val)
                    
                    #save reward time graph
                    reward_time(cum_rewards_val, name)
                    
                    #save reward price graph
                    reward_price(cum_rewards_val, cum_price_val, name)
                    
                    #save action price graph
                    action_price(action_0_val, action_1_val, action_2_val,cum_price_val, name)
                    
                    #save buy price graph
                    buysell_val = buysell_price(cum_price_val, cum_buysell_val)
                    buy_price(buysell_val, name)
                    
                    #Backtesting
                    df_each_trade_pnl = model_strategy(buysell_val, cost_percent, name, training_episode, num_parameter)
                    
                    #save return and underlying graph
                    df_cum_price_val = cum_perform(cum_price_val)
                    return_underlying(df_each_trade_pnl, df_cum_price_val, name)
                    
                    #corrleation anaylysis
                    construct_corr(cum_rewards_val, 
                                   cum_actions_val, 
                                   cum_buysell_val, 
                                   cum_price_val, 
                                   action_0_val, 
                                   action_1_val, 
                                   action_2_val, 
                                   name)
                    print('Finish' + name)