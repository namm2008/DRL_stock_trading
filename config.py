#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: matthewyeung
"""

import math
import random
import numpy as np
import pandas as pd
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from attention_gru import *
from attention_lstm import *
from replay_memory import *



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class hyperpara():
    def __init__(self, memory, policy_net, target_net, Transition, timestep, optimizer):
        self.memory = memory
        self.policy_net = policy_net
        self.target_net = target_net
        self.Transition = Transition
        self.timestep = timestep
        self.optimizer = optimizer
 
    #for gru with attention 
    def select_action(self, state, en_hidden_state, de_hidden_state):
        global steps_done, EPS_START, EPS_END, EPS_DECAY
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                action1, _ ,_ ,_ = self.policy_net(state.unsqueeze(0), en_hidden_state, de_hidden_state)
                action1 = action1.view(-1)
                if len(self.memory) == 0 or self.memory.last_ls() == 0:
                    mask = [True, True, False]
                    action = action1[mask].argmax().view(1,1)
                elif self.memory.last_ls() == 1:
                    mask = [True, False, True]
                    action = action1[mask].argmax().view(1,1)
                    if action == 1:
                        action = torch.tensor([2], dtype = torch.int64, device= device).view(1,1)
            return action
                                
        else:
            if len(self.memory) == 0 or self.memory.last_ls() == 0:
                return torch.tensor(random.randrange(2), device=device, dtype=torch.int64).view(1,1)
            elif self.memory.last_ls() == 1:
                return torch.tensor(random.randrange(2)*2, device=device, dtype=torch.int64).view(1,1)

        
    def select_action_val(self, state, en_hidden_state, de_hidden_state):
        with torch.no_grad():
            action1, _ ,_ ,_ = self.policy_net(state.unsqueeze(0), en_hidden_state, de_hidden_state)
            action1 = action1.view(-1)
            if len(self.memory) == 0 or self.memory.last_ls() == 0:
                mask = [True, True, False]
                action = action1[mask].argmax().view(1,1)
            elif self.memory.last_ls() == 1:
                mask = [True, False, True]
                action = action1[mask].argmax().view(1,1)
                if action == 1:
                    action = torch.tensor([2], dtype = torch.int64, device= device).view(1,1)
        return action, action1[0], action1[1], action1[2]


    def reward_action_train(self, dataloader, action, steps, cost_percent):
        #mean of the next 20 steps
        p_t1 = dataloader[self.timestep + steps : self.timestep + steps + 5].mean()
        p_t = dataloader[self.timestep + steps - 1]
        buy = p_t1 - p_t
        cost = p_t * cost_percent/100
    
        #for cash on hand:
        if len(self.memory) == 0 or self.memory.last_ls() == 0:
            if action == 0:
                rewards = 0
            elif action == 1:
                rewards = buy - cost
    
        #for holding stock:
        elif self.memory.last_ls() == 1:
            if action == 0:
                rewards = 0
            elif action == 2:
                rewards = buy
            
        return torch.tensor(rewards, dtype = torch.float32, device= device)  
    
    #for GRU with attention but without proritized replay
    def optimize_model_noper(self, GAMMA, BATCH_SIZE):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        
        # Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).view(BATCH_SIZE, self.timestep, -1)
        state_batch = torch.cat(batch.state).view(BATCH_SIZE, self.timestep, -1)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        global en_hidden, en_hidden_1, de_hidden, de_hidden_1, losss
        
        state_action, en_hidden, de_hidden, alpha = self.policy_net(state_batch, en_hidden, de_hidden)
        state_action_values = state_action.gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state , en_hidden_1, de_hidden_1, alpha_1 = self.target_net(non_final_next_states, en_hidden_1, de_hidden_1)
        next_state_values[non_final_mask] = next_state.max(1)[0].detach()
        next_state_values = next_state_values.view(BATCH_SIZE,1)
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
        # Compute loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        losss = loss.item()
    
    #for GRU with proritized replay and attention
    def optimize_model_per(self,GAMMA, BATCH_SIZE):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions, ids, weights = self.memory.sample(BATCH_SIZE)
        
        # Transition of batch-arrays.
        batch = self.Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None]).view(BATCH_SIZE, self.timestep, -1)
        state_batch = torch.cat(batch.state).view(BATCH_SIZE, self.timestep, -1)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        global en_hidden, en_hidden_1, de_hidden, de_hidden_1, losss
        
        state_action, en_hidden, de_hidden, alpha = self.policy_net(state_batch, en_hidden, de_hidden)
        state_action_values = state_action.gather(1, action_batch)
        
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state , en_hidden_1, de_hidden_1, alpha_1 = self.target_net(non_final_next_states, en_hidden_1, de_hidden_1)
        next_state_values[non_final_mask] = next_state.max(1)[0].detach()
        next_state_values = next_state_values.view(BATCH_SIZE,1)
        
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
        # Compute loss
        diff = state_action_values - expected_state_action_values
        loss = (0.5 * (diff * diff) * weights).mean()
    
        # Update memory
        delta = diff.abs().detach().cpu().numpy().tolist()
        self.memory.update_priorities(ids, delta)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        losss = loss.item()