# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:04:02 2023

@author: nilah
code adapted from jorge de huevels

d_model= 512? or 64?
"""

from __future__ import print_function
import logging


import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import _get_activation_fn
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_

from sb3_rl.feature_extractors.common.utils import tensor_from_observations_dict, create_mask_float, init_weights_xavier
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp

from sb3_rl.feature_extractors.common.positional_encoding import PositionalEncoding
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
        
        #additional 
        if self.config['NB_sensor_channels']==126:
            self.input_dim = 126
        self.output_dim = self.config['num_classes']
        self.window_size = self.config['sliding_window_length']
        self.transformer_dim = transformer_dim if n_embedding_layers > 0 else self.input_dim
        self.dim_fc = dim_fc
        self.dim_fully_connected = dim_fully_connected
        self.n_layers = n_layers
        self.n_embedding_layers = n_embedding_layers

        self.multihead_attn = MultiheadAttention( n_head, d_model=self.dim_fully_connected, batch_first=batch_first, dropout= 0.1)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward,)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        
        
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            memory_mask: the mask for the memory sequence (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
