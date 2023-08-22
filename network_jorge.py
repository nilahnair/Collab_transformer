# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:04:02 2023

@author: nilah
code adapted from jorge de huevels

d_model= 512? or 64?
"""

from __future__ import print_function
import logging


import torch as th
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from torch.nn.init import xavier_uniform_

#from sb3_rl.feature_extractors.common.utils import tensor_from_observations_dict, create_mask_float, init_weights_xavier
#from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp

#from sb3_rl.feature_extractors.common.positional_encoding import PositionalEncoding
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer



import numpy as np


class Network(nn.Module):
    '''
    classdocs
    '''


    def __init__(self, config):
        '''
        Constructor
        '''
        
        super(Network, self).__init__()
        
        logging.info('            Network: Constructor')
        
        self.config = config
        transformer_dim=64
        n_head=8
        dim_fully_connected=64
        dim_fc=128
        n_layers=6
        n_embedding_layers=4
        use_pos_embedding=True
        activation_function='relu'
        norm_first = True
        layer_norm_eps = 1e-5
        
        #not sure of these values
        reverse_temporal_encoding = False
        fc_input_arch_list = [[32], [16], [16]]
        embedding_bias= False
        
        #additional 
        if self.config['NB_sensor_channels']==126:
            self.input_dim = 126
        self.output_dim = self.config['num_classes']
        self.window_size = self.config['sliding_window_length']
        self.transformer_dim = transformer_dim if n_embedding_layers > 0 else self.input_dim
        d_model= dim_fully_connected
        self.n_heads= n_head
        self.dim_fc = dim_fc
        self.n_layers = n_layers
        self.n_embedding_layers = n_embedding_layers
        self.temporal_encoding_type = "wave"
        mlp_embedding: bool = True
        self.activation_function = nn.ReLU()
        self.norm_first = norm_first
        self.dropout=0.1
        
        
        # Positional encoding (temporal)
        if self.temporal_encoding_type == "single":
            #self.temporal_encoding_cached = th.exp(-1 * torch.arange(self.dim_temporal).to(get_device("auto")))  # TODO
            if reverse_temporal_encoding:
                self.temporal_encoding_cached = th.flip(self.temporal_encoding_cached, dims=[-1])
            self.positional_encoding = self.add_time_encoding_to_tensor

        elif self.temporal_encoding_type == "wave":
            self.positional_encoding = Summer(PositionalEncoding1D(self.transformer_dim))
            
        #input embedding
        self.input_proj = nn.ModuleList()
        for _ in range(self.n_embedding_layers):
            d_in = self.input_dim if len(self.input_proj) == 0 else self.transformer_dim
            mlp_layer = nn.Sequential(nn.Linear(d_in, self.transformer_dim), nn.ReLU(),
                                      nn.Linear(self.transformer_dim, self.transformer_dim), nn.ReLU())
            self.input_proj.append(mlp_layer)
            
        
        # TRANSFORMER ENCODER
        # ===================================================================================================
        transformer_encoder = th.nn.TransformerEncoderLayer(
            d_model=self.transformer_dim,  # TODO Check
            nhead=self.n_heads,
            dim_feedforward=self.dim_fc,
            norm_first=self.norm_first,
            batch_first=True,
            dropout=self.dropout,
        )
        self.transformer_encoder = th.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=self.n_layers,
        )
        
         #setting mlp layers
        self.imu_head = nn.Sequential(nn.LayerNorm(self.transformer_dim), nn.Linear(self.transformer_dim, self.transformer_dim//4),
                                      self.activation_function, nn.Dropout(0.1), nn.Linear(self.transformer_dim//4, self.output_dim))
        
        
        #self.transformer_encoder.apply(init_weights_xavier)
        #nn.init.xavier_normal(self.transformer_encoder.weight)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        
       #is inout [B,Win,D] then reshape to [B,D,Win]
        #x = x.transpose(1, 2)
        #here[B,1,Win,D] to [B,Win,D,1]
        x=x.permute(0,2,3,1)
        #to [B,D,Win]
        x = x.view(x.size()[0], x.size()[1], x.size()[2])
        #to [B,Win,D]
        #x=x.permute(0,2,1)
        #print('before getting embedding')
        #print(x.shape)
        #input embedding
        for mlp_layer in self.input_proj:
            x = mlp_layer(x)
        #print('after getting embedding')
        #print(x.shape)
        #Reshaping: [B, D', Win] -> [Win, B, D'] 
        x = x.permute(2, 0, 1)
        
        # Prepend class token: [Win, B, D']  -> [Win+1, B, D']
        #cls_token = self.cls_token.unsqueeze(1).repeat(1, x.shape[1], 1)
        #x = th.cat([cls_token, x])
        
        #position embedding
        if self.temporal_encoding_type == "wave":
            x = self.positional_encoding(x)
            
        #transformer
        # Transformer Encoder pass
        x = self.transformer_encoder(x)[0]
        
        x= self.imu_head(x)
    
        if not self.training:
            if self.config['output'] == 'softmax':
                x = self.softmax(x)
                    
        return x