#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:55:04 2021

@author: matthewyeung
"""
import math
import random
import numpy as np
import pandas as pd

from collections import namedtuple
from itertools import count

from IPython.display import clear_output
from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#for GRU with attention but without proritized replay
def optimize_model_noper():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    
    # Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).view(BATCH_SIZE, timestep, -1)
    state_batch = torch.cat(batch.state).view(BATCH_SIZE, timestep, -1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    global en_hidden, en_hidden_1, de_hidden, de_hidden_1, losss
    
    state_action, en_hidden, de_hidden, alpha = policy_net(state_batch, en_hidden, de_hidden)
    state_action_values = state_action.gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state , en_hidden_1, de_hidden_1, alpha_1 = target_net(non_final_next_states, en_hidden_1, de_hidden_1)
    next_state_values[non_final_mask] = next_state.max(1)[0].detach()
    next_state_values = next_state_values.view(BATCH_SIZE,1)
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    losss = loss.item()

#for GRU with proritized replay and attention
def optimize_model_per():
    if len(memory) < BATCH_SIZE:
        return
    transitions, ids, weights = memory.sample(BATCH_SIZE)
    
    # Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None]).view(BATCH_SIZE, timestep, -1)
    state_batch = torch.cat(batch.state).view(BATCH_SIZE, timestep, -1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    global en_hidden, en_hidden_1, de_hidden, de_hidden_1, losss
    
    state_action, en_hidden, de_hidden, alpha = policy_net(state_batch, en_hidden, de_hidden)
    state_action_values = state_action.gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state , en_hidden_1, de_hidden_1, alpha_1 = target_net(non_final_next_states, en_hidden_1, de_hidden_1)
    next_state_values[non_final_mask] = next_state.max(1)[0].detach()
    next_state_values = next_state_values.view(BATCH_SIZE,1)
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute loss
    diff = state_action_values - expected_state_action_values
    loss = (0.5 * (diff * diff) * weights).mean()

    # Update memory
    delta = diff.abs().detach().cpu().numpy().tolist()
    memory.update_priorities(ids, delta)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    losss = loss.item()