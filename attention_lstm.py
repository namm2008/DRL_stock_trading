#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:02:08 2021

@author: matthewyeung
"""

import math
import random
import numpy as np
import pandas as pd

from itertools import count
from PIL import Image
from IPython.display import clear_output
from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#LSTM with attention model
class LSTM_Attention(nn.Module):
  
    def __init__(self, feature_number, n_layers, seq_length, encoder_hidden_dim, decoder_hidden_dim, output_size, 
                 bidirectional = True, drop_prob=0.5):
        super(LSTM_Attention, self).__init__()
        self.n_layers = n_layers
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.output_size = output_size
        self.seq_length = seq_length
        self.bidirectional = bidirectional
        
        self.encoder_output_dim = encoder_hidden_dim*(1+ self.bidirectional)
        
        
        self.lstm_encoder= nn.LSTM(feature_number, encoder_hidden_dim, n_layers, 
                         dropout=drop_prob, bidirectional=bidirectional, batch_first=True)           
        self.attn = nn.Linear(self.encoder_output_dim + decoder_hidden_dim, 1)
        self.lstm = nn.LSTM(self.encoder_output_dim + feature_number, decoder_hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc1 = nn.Linear(decoder_hidden_dim*seq_length, 8)
        self.fc2 = nn.Linear(8, output_size)
        self.softmax = nn.Softmax(dim=1)
  
    def init_hidden_encoder(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers*(1 + self.bidirectional), batch_size, self.encoder_hidden_dim).zero_().to(device),
                  weight.new(self.n_layers*(1 + self.bidirectional), batch_size, self.encoder_hidden_dim).zero_().to(device))
        
                      
        return hidden
    
    def init_hidden_decoder(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(1, batch_size, self.decoder_hidden_dim).zero_().to(device),
                  weight.new(1, batch_size, self.decoder_hidden_dim).zero_().to(device))#(1,64,13)
        return hidden
  
    def forward(self, x, encoder_hidden, decoder_hidden):
        batch_size = x.size(0)

        encoder_outputs, encoder_hidden = self.lstm_encoder(x, encoder_hidden) 
        #encoder_outputs shape = (64, 20, 50)(batch, seq_len, hidden*(1+bidirect))
        
        encoder_outputs = encoder_outputs.permute(1,0,2)
        
        weights = []
        for i in range(len(encoder_outputs)):
            concat = torch.cat((decoder_hidden[0][0], encoder_outputs[i]), dim = 1)
            attn_ = self.attn(concat)
            weights.append(attn_)
        
        alpha = F.softmax(torch.cat(weights, 1), 1)
        
        attn_applied = alpha.unsqueeze(2)*encoder_outputs.permute(1,0,2) # (64,20,50)
        
        input_lstm = torch.cat((attn_applied, x), dim = 2) #(64, 20, 57)
        
        lstm_out, decoder_hidden = self.lstm(input_lstm, decoder_hidden)
        
        lstm_out = lstm_out.contiguous().view(batch_size, -1)
        
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.softmax(self.fc2(out))

        return out, encoder_hidden, decoder_hidden, alpha