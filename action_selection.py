#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:50:24 2021

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

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#for gru with attention 
def select_action(state, en_hidden_state, de_hidden_state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            action1, _ ,_ ,_ = policy_net(state.unsqueeze(0), en_hidden_state, de_hidden_state)
            action1 = action1.view(-1)
            if len(memory) == 0 or memory.last_ls() == 0:
                mask = [True, True, False]
                action = action1[mask].argmax().view(1,1)
            elif memory.last_ls() == 1:
                mask = [True, False, True]
                action = action1[mask].argmax().view(1,1)
                if action == 1:
                    action = torch.tensor([2], dtype = torch.int64, device= device).view(1,1)
        return action
                            
    else:
        if len(memory) == 0 or memory.last_ls() == 0:
            return torch.tensor(random.randrange(2), device=device, dtype=torch.int64).view(1,1)
        elif memory.last_ls() == 1:
            return torch.tensor(random.randrange(2)*2, device=device, dtype=torch.int64).view(1,1)

        
def select_action_val(state, en_hidden_state, de_hidden_state):
    with torch.no_grad():
        action1, _ ,_ ,_ = policy_net(state.unsqueeze(0), en_hidden_state, de_hidden_state)
        action1 = action1.view(-1)
        if len(memory) == 0 or memory.last_ls() == 0:
            mask = [True, True, False]
            action = action1[mask].argmax().view(1,1)
        elif memory.last_ls() == 1:
            mask = [True, False, True]
            action = action1[mask].argmax().view(1,1)
            if action == 1:
                action = torch.tensor([2], dtype = torch.int64, device= device).view(1,1)
    return action, action1[0], action1[1], action1[2]


def reward_action_train(dataloader, action, steps, timestep, cost_percent):
    #mean of the next 20 steps
    p_t1 = dataloader[timestep + steps : timestep + steps + 5].mean()
    p_t = dataloader[timestep + steps - 1]
    buy = p_t1 - p_t
    cost = p_t * cost_percent/100

    #for cash on hand:
    if len(memory) == 0 or memory.last_ls() == 0:
        if action == 0:
            rewards = 0
        elif action == 1:
            rewards = buy - cost

    #for holding stock:
    elif memory.last_ls() == 1:
        if action == 0:
            rewards = 0
        elif action == 2:
            rewards = buy
        
    return torch.tensor(rewards, dtype = torch.float32, device= device)