#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:42:56 2021

@author: matthewyeung
"""
import math
import random
import numpy as np
import pandas as pd
from collections import namedtuple


import torch



# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#without using pytorch Dataloader module, use the dataset itself
def state_loader(dataset, timestep, start_from):
    dataset = dataset[start_from:start_from + timestep, :]
    return dataset

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'longshort'))

# Random Experience Replay Memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    #Save new transition
    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.memory[-1] = transition
        if self.position >= self.capacity:
            del self.memory[0]
            self.memory.append(None)
            self.memory[-1] = transition
        self.position += 1
    
    #Sampling
    def sample(self, batch_size):
        
        self.sample_list = []
        self.sample_list = random.sample(self.memory, batch_size)
        
        return self.sample_list
        
    def last_ls(self):
        return self.memory[-1][-1]

    def __len__(self):
        return len(self.memory)

    
    
# Prioritized Experience Replay    
class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=10000):
        self.prob_alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0**self.prob_alpha

        total = len(self.buffer)
        if total < self.capacity:
            pos = total
            self.buffer.append(transition)
        else:
            prios = self.priorities[:total]
            probs = (1 - prios / prios.sum()) / (total - 1)
            pos = np.random.choice(total, 1, p=probs)

        self.priorities[pos] = max_prio

    def sample(self, batch_size):
        total = len(self.buffer)
        prios = self.priorities[:total]
        probs = prios / prios.sum()

        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # Min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min*total)**(-beta)

        weights = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        weights = torch.tensor(weights, device=device, dtype=torch.float)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio[0] + 1e-5)**self.prob_alpha
    
    def last_ls(self):
        return self.buffer[-1][-1]

    def __len__(self):
        return len(self.buffer)