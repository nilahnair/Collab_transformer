# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:20:25 2023

@author: nilah

adaptation of raphael's network to the code structure of fernando moya
"""

from __future__ import print_function
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import numpy as np


class Network(nn.Module):
    '''
    classdocs
    '''


    def __init__(self, config):
        '''
        Constructor
        
        update config with input_dim, output_dim, transformer_dim=64, n_head=8, dim_fc=128, n_layers=6, 
        n_embedding_layers=4, use_pos_embedding=true, activation_function='gelu'?
        '''
        
        super(Network, self).__init__()
        
        logging.info('            Network: Constructor')
        
        self.config = config
        transformer_dim=64
        n_head=8
        dim_fc=128
        n_layers=6
        n_embedding_layers=4
        use_pos_embedding=True
        activation_function='gelu'
        
        #additional 
        if self.config['NB_sensor_channels']==126:
            self.input_dim = 126
        self.output_dim = self.config['num_classes']
        self.window_size = self.config['sliding_window_length']
        self.transformer_dim = transformer_dim if n_embedding_layers > 0 else self.input_dim
        self.n_head = get_nhead(self.transformer_dim, n_head)
        self.dim_fc = dim_fc
        self.n_layers = n_layers
        self.n_embedding_layers = n_embedding_layers
        self.use_pos_embedding = use_pos_embedding
        self.activation_function = nn.GELU() if activation_function.lower() == 'gelu' else nn.ReLU()
        
        #input embedding definition
        self.input_proj = nn.ModuleList()
        for _ in range(self.n_embedding_layers):
            d_in = self.input_dim if len(self.input_proj) == 0 else self.transformer_dim
            conv_layer = nn.Sequential(nn.Conv1d(d_in, self.transformer_dim, 1), self.activation_function)
            self.input_proj.append(conv_layer)
            
        #setting parameters
        self.cls_token = nn.Parameter(torch.zeros((1, self.transformer_dim)))
        
        #setting positional encoding
        if self.use_pos_embedding:
            self.position_embed = nn.Parameter(torch.randn(self.window_size + 1, 1, self.transformer_dim))
        
        #set transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model = self.transformer_dim, nhead = self.n_head, dim_feedforward = self.dim_fc,
                                       dropout = 0.1, activation = 'gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers = self.n_layers, norm = nn.LayerNorm(self.transformer_dim))
        
        #setting mlp layers
        self.imu_head = nn.Sequential(nn.LayerNorm(self.transformer_dim), nn.Linear(self.transformer_dim, self.transformer_dim//4),
                                      self.activation_function, nn.Dropout(0.1), nn.Linear(self.transformer_dim//4, self.output_dim))
        
        self.softmax = nn.Softmax()
        
        # initialisation of parameters
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return
    
    def __str__(self):
        return 'Transformer_Encoder with dim={} & activation={}'.format(self.transformer_dim, str(self.activation_function))
    
     
    def forward(self, x):
        '''
        Forwards function, required by torch.

        @param x: batch [batch, 1, Channels, Time], Channels = Sensors * 3 Axis
        B - batch,  win - window length and D - input dim (at the start) after embedding 64
        @return x: Output of the network, either Softmax or Attribute
        '''
        #is inout [B,Win,D] then reshape to [B,D,Win]
        #x = x.transpose(1, 2)
        #here[B,1,D,Win] to [B,D,Win,1]
        print('shape before permutation')
        print(x.shape)
        x=x.permute(0,2,3,1)
        print('shape after permutation')
        print(x.shape)
        #to [B,D,Win]
        x = x.view(x.size()[0], x.size()[1], x.size()[2])
        
        #input embedding
        for conv_layer in self.input_proj:
            x = conv_layer(x)
            
        # Reshaping: [B, D', Win] -> [Win, B, D'] 
        x = x.permute(2, 0, 1)
        
        # Prepend class token: [Win, B, D']  -> [Win+1, B, D']
        cls_token = self.cls_token.unsqueeze(1).repeat(1, x.shape[1], 1)
        x = torch.cat([cls_token, x])
        
        #position embedding
        if self.use_pos_embedding:
            x += self.position_embed
            
        #transformer
        # Transformer Encoder pass
        x = self.transformer_encoder(x)[0]
        
        '''
        if self.config["fully_convolutional"] == "FC":
            x = F.dropout(x, training=self.training)
            x = F.relu(self.fc4(x))
            x = F.dropout(x, training=self.training)
            x = self.fc5(x)
        '''
            
        # Pass through fully-connected layers
        x= self.imu_head(x)
    
        if not self.training:
            if self.config['output'] == 'softmax':
                x = self.softmax(x)

        return x
    
    
def get_nhead(embed_dim, n_head):
    for hd in range(n_head, 0, -1):
        if embed_dim % hd == 0:
            logging.info('N_head = {}'.format(hd))
            return hd
        