# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import warnings
import torch
import torch.nn as nn
from .pos_embed import rope_apply_multires as rope_apply

try:
    from flash_attn import (flash_attn_varlen_func)
    FLASHATTN_IS_AVAILABLE = True
except ImportError as e:
    FLASHATTN_IS_AVAILABLE = False
    flash_attn_varlen_func = None
    warnings.warn(f'{e}')

__all__ = [
    "drop_path",
    "modulate",
    "PatchEmbed",
    "DropPath",
    "RMSNorm",
    "Mlp",
    "TimestepEmbedder",
    "DiTEditBlock",
    "MultiHeadAttentionDiTEdit",
    "T2IFinalLayer",
]

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def modulate(x, shift, scale, unsqueeze=False):
    if unsqueeze:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        return x * (1 + scale) + shift
    

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) *
            torch.arange(start=0, end=half, dtype=torch.float32) /
            half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class DiTACEBlock(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_heads,
                 mlp_ratio=4.0,
                 drop_path=0.,
                 window_size=0,
                 backend=None,
                 use_condition=True,
                 qk_norm=False,
                 **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_condition = use_condition
        self.norm1 = nn.LayerNorm(hidden_size,
                                  elementwise_affine=False,
                                  eps=1e-6)
        self.attn = MultiHeadAttention(hidden_size,
                                        num_heads=num_heads,
                                        qkv_bias=True,
                                        backend=backend,
                                        qk_norm=qk_norm,
                                        **block_kwargs)
        if self.use_condition:
            self.cross_attn = MultiHeadAttention(
                hidden_size,
                context_dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                backend=backend,
                qk_norm=qk_norm,
                **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size,
                                  elementwise_affine=False,
                                  eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate='tanh')
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=approx_gelu,
                       drop=0)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.window_size = window_size
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5)

    def forward(self, x, y, t, **kwargs):
        B = x.size(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(1), scale_msa.squeeze(1), gate_msa.squeeze(1),
            shift_mlp.squeeze(1), scale_mlp.squeeze(1), gate_mlp.squeeze(1))
        x = x + self.drop_path(gate_msa * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa, unsqueeze=False), **
            kwargs))
        if self.use_condition:
            x = x + self.cross_attn(x, context=y, **kwargs)

        x = x + self.drop_path(gate_mlp * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp, unsqueeze=False)))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 dim,
                 context_dim=None,
                 num_heads=None,
                 head_dim=None,
                 attn_drop=0.0,
                 qkv_bias=False,
                 dropout=0.0,
                 backend=None,
                 qk_norm=False,
                 eps=1e-6,
                 **block_kwargs):
        super().__init__()
        # consider head_dim first, then num_heads
        num_heads = dim // head_dim if head_dim else num_heads
        head_dim = dim // num_heads
        assert num_heads * head_dim == dim
        context_dim = context_dim or dim
        self.dim = dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = math.pow(head_dim, -0.25)
        # layers
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.v = nn.Linear(context_dim, dim, bias=qkv_bias)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = RMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.attention_op = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.backend = backend
        assert self.backend in ('flash_attn', 'xformer_attn', 'pytorch_attn',
                                None)
        if FLASHATTN_IS_AVAILABLE and self.backend in ('flash_attn', None):
            self.backend = 'flash_attn'
            self.softmax_scale = block_kwargs.get('softmax_scale', None)
            self.causal = block_kwargs.get('causal', False)
            self.window_size = block_kwargs.get('window_size', (-1, -1))
            self.deterministic = block_kwargs.get('deterministic', False)
        else:
            raise NotImplementedError

    def flash_attn(self, x, context=None, **kwargs):
        '''
         The implementation will be very slow when mask is not None,
         because we need rearange the x/context features according to mask.
        Args:
            x:
            context:
            mask:
            **kwargs:
        Returns: x
        '''
        dtype = kwargs.get('dtype', torch.float16)

        def half(x):
            return x if x.dtype in [torch.float16, torch.bfloat16
                                    ] else x.to(dtype)

        x_shapes = kwargs['x_shapes']
        freqs = kwargs['freqs']
        self_x_len = kwargs['self_x_len']
        cross_x_len = kwargs['cross_x_len']
        txt_lens = kwargs['txt_lens']
        n, d = self.num_heads, self.head_dim

        if context is None:
            # self-attn
            q = self.norm_q(self.q(x)).view(-1, n, d)
            k = self.norm_q(self.k(x)).view(-1, n, d)
            v = self.v(x).view(-1, n, d)
            q = rope_apply(q, self_x_len, x_shapes, freqs, pad=False)
            k = rope_apply(k, self_x_len, x_shapes, freqs, pad=False)
            q_lens = k_lens = self_x_len
        else:
            # cross-attn
            q = self.norm_q(self.q(x)).view(-1, n, d)
            k = self.norm_q(self.k(context)).view(-1, n, d)
            v = self.v(context).view(-1, n, d)
            q_lens = cross_x_len
            k_lens = txt_lens

        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]),
                                  q_lens]).cumsum(0, dtype=torch.int32)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]),
                                  k_lens]).cumsum(0, dtype=torch.int32)
        max_seqlen_q = q_lens.max()
        max_seqlen_k = k_lens.max()

        out_dtype = q.dtype
        q, k, v = half(q), half(k), half(v)
        x = flash_attn_varlen_func(q,
                                   k,
                                   v,
                                   cu_seqlens_q=cu_seqlens_q,
                                   cu_seqlens_k=cu_seqlens_k,
                                   max_seqlen_q=max_seqlen_q,
                                   max_seqlen_k=max_seqlen_k,
                                   dropout_p=self.attn_drop.p,
                                   softmax_scale=self.softmax_scale,
                                   causal=self.causal,
                                   window_size=self.window_size,
                                   deterministic=self.deterministic)

        x = x.type(out_dtype)
        x = x.reshape(-1, n * d)
        x = self.o(x)
        x = self.dropout(x)
        return x

    def forward(self, x, context=None, **kwargs):
        x = getattr(self, self.backend)(x, context=context, **kwargs)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size,
                                       elementwise_affine=False,
                                       eps=1e-6)
        self.linear = nn.Linear(hidden_size,
                                patch_size * patch_size * out_channels,
                                bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2,
                                                                         dim=1)
        shift, scale = shift.squeeze(1), scale.squeeze(1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x