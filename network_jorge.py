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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp

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
        activation_function='gelu'
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
        self.dim_fc = dim_fc
        self.dim_fully_connected = dim_fully_connected
        self.n_layers = n_layers
        self.n_embedding_layers = n_embedding_layers
        self.temporal_encoding_type = "wave"
        mlp_embedding: bool = False
        
        
        # Positional encoding (temporal)
        if self.temporal_encoding_type == "single":
            #self.temporal_encoding_cached = th.exp(-1 * torch.arange(self.dim_temporal).to(get_device("auto")))  # TODO
            if reverse_temporal_encoding:
                self.temporal_encoding_cached = th.flip(self.temporal_encoding_cached, dims=[-1])
            self.positional_encoding = self.add_time_encoding_to_tensor

        elif self.temporal_encoding_type == "wave":
            self.positional_encoding = Summer(PositionalEncoding1D(self.dim_fully_connected))
            
        #input embedding
        # FC LAYER NETWORKS - INDIVIDUAL
        # ===================================================================================================
        self.mlp_embedding = [mlp_embedding] * len(fc_input_arch_list) if isinstance(mlp_embedding, bool) else mlp_embedding
        self.fc_input_embedding = th.nn.ModuleList()
        for a, architecture in enumerate(fc_input_arch_list):
            if architecture[0] > 0:
                if self.mlp_embedding[a]:
                    fc_input_embedding = create_mlp(self.input_dim[a], 0, architecture, th.nn.ReLU, squash_output=False)
                else:
                    fc_input_embedding = [th.nn.Linear(self.input_dim[a], architecture[0], bias=embedding_bias)]
            else:
                fc_input_embedding = [th.nn.Identity()]
            self.fc_input_embedding.append(th.nn.Sequential(*fc_input_embedding))

        # FC LAYER NETWORKS - CONCATENATED
        # ===================================================================================================
        if mlp_embedding:
            fc_concat_embedding = create_mlp(
                input_dim=self.dim_merged_embedded,
                output_dim=0,
                net_arch=[self.dim_fully_connected],
                activation_fn=th.nn.ReLU,
                squash_output=False,
            )
        else:
            fc_concat_embedding = [th.nn.Linear(self.dim_merged_embedded, self.dim_fully_connected, bias=embedding_bias)]
        self.fc_concat_embedding = th.nn.Sequential(*fc_concat_embedding)
        
        # TRANSFORMER ENCODER
        # ===================================================================================================
        transformer_encoder = th.nn.TransformerEncoderLayer(
            d_model=self.dim_fully_connected,  # TODO Check
            nhead=self.num_transformer_heads,
            dim_feedforward=self.dim_transformer_feedforward,
            norm_first=self.norm_first,
            batch_first=True,
            dropout=self.dropout,
        )
        self.transformer_encoder = th.nn.TransformerEncoder(
            encoder_layer=transformer_encoder,
            num_layers=self.num_transformer_layers,
        )
        
        self.transformer_encoder.apply(init_weights_xavier)
        
        self.softmax = nn.Softmax()
        
    def forward(self,x):
        
      # OPTIONAL Positional Encoding "Single"
        # ====================================================================
        if self.temporal_encoding_type == "single":
            input[self.temporal_encoding_vec_ind] = self.self.positional_encoding(input[self.temporal_encoding_vec_ind])

        # Transformer Embedding
        # ====================================================================
        input_embedded = [fc_input_embedding(input[i]) for i, fc_input_embedding in enumerate(self.fc_input_embedding)]
        x = th.concatenate(input_embedded, dim=2)
        x = self.fc_concat_embedding(x)

        # OPTIONAL Positional Encoding "Wave"
        # ====================================================================
        if self.temporal_encoding_type == "wave":
            x = self.positional_encoding(x)

        # Transformer Encoder
        # ====================================================================
        x = self.transformer_encoder(
            src=x,
            mask=self.mask if self.mask_future else None,
            # is_causal=self.memory_is_causal,
        )
        
        if not self.training:
            if self.config['output'] == 'softmax':
                x = self.softmax(x)

        return x
