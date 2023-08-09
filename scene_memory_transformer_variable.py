from stable_baselines3.common.utils import get_device
from typing import Dict, List, Tuple, Type, Union
from torch.nn.modules.transformer import _get_activation_fn

from sb3_rl.feature_extractors.common.utils import tensor_from_observations_dict, create_mask_float, init_weights_xavier
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
import torch as th
import gym
from sb3_rl.feature_extractors.common.positional_encoding import PositionalEncoding
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

class SceneMemoryTransformerVariable(BaseFeaturesExtractor):
    """
    TODOs
    - Recompute positions with respect to the last position for every step
        --> Add position to task_obs so that we have a reference!
    """
    def __init__(
            self,
            observation_space: gym.spaces.Dict,
            dim_temporal: int,
            dim_feature_out: int = None,
            key_input_list: List[str] = ['scan', 'task_obs', 'action'],
            key_skip_list: List[str] = [],
            temporal_encoding_vec_ind: int = 1,
            temporal_encoding_type: str = 'single',
            reverse_temporal_encoding: bool = False,
            num_transformer_heads: int = 8,
            num_transformer_layers: int = 1,
            mask_future: bool = False,
            mask_future_decoder: bool = True,
            norm_first: bool = False,
            dropout: float = 0.0,
            fc_input_arch_list: List[List[int]] = [[32], [16], [16]],
            mlp_embedding: bool = False,
            embedding_bias: bool = False,
            dim_fully_connected: int = 64,
            dim_transformer_feedforward: int = None, # dim of inflation in middle of MLP
            latest_observation_first: bool = True,
            memory_is_causal: bool = False,
    ) -> None:

        # CHECKS
        # ===================================================================================================
        assert len(key_input_list) == len(fc_input_arch_list)

        # PARAMETER AND SETTINGS
        # ===================================================================================================
        self.key_input_list = key_input_list
        self.key_skip_ids = [self.key_input_list.index(key) for key in key_skip_list]

        # Everything temporal:
        self.temporal_encoding_vec_ind = temporal_encoding_vec_ind
        self.temporal_encoding_type = temporal_encoding_type
        self.dim_temporal = dim_temporal
        self.current_obs_ind = 0 if latest_observation_first else self.dim_temporal - 1

        # Calculate input dimensions:
        self.dim_input = list()
        for key in self.key_input_list:
            self.dim_input.append(observation_space[key].shape[-1])
        self.dim_input_orig = copy.copy(self.dim_input)
        if self.temporal_encoding_type == "single":
            self.dim_input[self.temporal_encoding_vec_ind] += 1
        self.dim_merged = sum(self.dim_input)
        self.dim_fully_connected = dim_fully_connected
        self.dim_transformer_feedforward = dim_transformer_feedforward if dim_transformer_feedforward is not None else dim_fully_connected
        self.dim_merged_embedded = 0
        for a, architecture in enumerate(fc_input_arch_list):
            if architecture[-1] > 0:
                self.dim_merged_embedded += architecture[-1]
            else:
                self.dim_merged_embedded += self.dim_input[a]

        # Calculate output dimension
        if dim_feature_out is None:
            dim_feature_out = self.dim_fully_connected
            for ind in self.key_skip_ids:
                dim_feature_out += self.dim_input[ind]

        super().__init__(observation_space=observation_space, features_dim=dim_feature_out)


        # Positional encoding (temporal)
        if self.temporal_encoding_type == "single":
            self.temporal_encoding_cached = torch.exp(-1 * torch.arange(self.dim_temporal).to(get_device("auto")))  # TODO
            if reverse_temporal_encoding:
                self.temporal_encoding_cached = th.flip(self.temporal_encoding_cached, dims=[-1])
            self.positional_encoding = self.add_time_encoding_to_tensor

        elif self.temporal_encoding_type == "wave":
            from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, \
                PositionalEncoding3D, Summer
            self.positional_encoding = Summer(PositionalEncoding1D(self.dim_fully_connected))


        # Everything transformer related
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layers = num_transformer_layers
        self.mask_future = mask_future
        self.mask_future_decoder = mask_future_decoder
        self.mask = create_mask_float(self.dim_temporal)
        self.memory_is_causal = memory_is_causal
        self.norm_first = norm_first
        self.dropout = dropout

        # FC LAYER NETWORKS - INDIVIDUAL
        # ===================================================================================================
        self.mlp_embedding = [mlp_embedding] * len(fc_input_arch_list) if isinstance(mlp_embedding, bool) else mlp_embedding
        self.fc_input_embedding = th.nn.ModuleList()
        for a, architecture in enumerate(fc_input_arch_list):
            if architecture[0] > 0:
                if self.mlp_embedding[a]:
                    fc_input_embedding = create_mlp(self.dim_input[a], 0, architecture, th.nn.ReLU, squash_output=False)
                else:
                    fc_input_embedding = [th.nn.Linear(self.dim_input[a], architecture[0], bias=embedding_bias)]
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

        # TRANSFORMER DECODER
        # ===================================================================================================
        self.transformer_decoder = SMTTransformerDecoderLayer(
            d_model=self.dim_fully_connected,  # TODO Check
            nhead=self.num_transformer_heads,
            dim_feedforward=self.dim_transformer_feedforward,
            norm_first=self.norm_first,
            batch_first=True,
            dropout=self.dropout,
        )

        self.transformer_encoder.apply(init_weights_xavier)
        self.transformer_decoder.apply(init_weights_xavier)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Preparations
        # ====================================================================
        input = tensor_from_observations_dict(
            observations,
            self.key_input_list,
            self.dim_temporal,
            self.dim_input_orig
        )

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
        context = self.transformer_encoder(
            src=x,
            mask=self.mask if self.mask_future else None,
            # is_causal=self.memory_is_causal,
        )

        # Transformer Decoder
        # ====================================================================
        current_obs = torch.index_select(x, -2, torch.tensor([self.current_obs_ind, ], device=get_device("auto")))
        x = self.transformer_decoder(
            tgt=current_obs,
            memory=context,
            memory_mask=self.mask if self.mask_future_decoder else None,
            # memory_is_causal=self.memory_is_causal,  # TODO Check
        )
        x = th.flatten(x, start_dim=1)

        # Skip connections around transformer block of the non-embedded inputs
        # ====================================================================
        if self.key_skip_ids:
            skip_current_obs = [
                torch.index_select(
                    input[ind],
                    -2,
                    torch.tensor([self.current_obs_ind, ], device=get_device("auto"))
                ) for ind in self.key_skip_ids
            ]

            skip_current_obs = torch.concatenate(skip_current_obs, dim=-1)
            skip_current_obs = th.flatten(skip_current_obs, start_dim=1)
            x = torch.concatenate([x, skip_current_obs], dim=-1)

        return x

    def add_time_encoding_to_tensor(self, x):
        te = self.temporal_encoding_cached
        te_reshaped = te.view((1,) + (te.shape[0],) + (1,) * (x.ndim - 2))
        te_reshaped = th.repeat_interleave(te_reshaped, x.shape[0], dim=0)
        x = torch.concatenate([x, te_reshaped], dim=-1)
        return x


class SMTTransformerDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

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
