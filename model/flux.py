# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from collections import OrderedDict
from functools import partial
import warnings
from contextlib import nullcontext
import torch
from einops import rearrange, repeat
from scepter.modules.model.base_model import BaseModel
from scepter.modules.model.registry import BACKBONES
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint_sequential
import torch.nn.functional as F
import torch.utils.dlpack
import transformers
from scepter.modules.model.embedder.base_embedder import BaseEmbedder
from scepter.modules.model.registry import EMBEDDERS
from scepter.modules.model.tokenizer.tokenizer_component import (
    basic_clean, canonicalize, heavy_clean, whitespace_clean)
try:
    from transformers import AutoTokenizer, T5EncoderModel
except Exception as e:
    warnings.warn(
        f'Import transformers error, please deal with this problem: {e}')

from .layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)



@EMBEDDERS.register_class()
class ACETextEmbedder(BaseEmbedder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    """
        Uses the OpenCLIP transformer encoder for text
        """
    para_dict = {
        'PRETRAINED_MODEL': {
            'value':
            'google/umt5-small',
            'description':
            'Pretrained Model for umt5, modelcard path or local path.'
        },
        'TOKENIZER_PATH': {
            'value': 'google/umt5-small',
            'description':
            'Tokenizer Path for umt5, modelcard path or local path.'
        },
        'FREEZE': {
            'value': True,
            'description': ''
        },
        'USE_GRAD': {
            'value': False,
            'description': 'Compute grad or not.'
        },
        'CLEAN': {
            'value':
            'whitespace',
            'description':
            'Set the clean strtegy for tokenizer, used when TOKENIZER_PATH is not None.'
        },
        'LAYER': {
            'value': 'last',
            'description': ''
        },
        'LEGACY': {
            'value':
            True,
            'description':
            'Whether use legacy returnd feature or not ,default True.'
        }
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        pretrained_path = cfg.get('PRETRAINED_MODEL', None)
        self.t5_dtype = cfg.get('T5_DTYPE', 'float32')
        assert pretrained_path
        with FS.get_dir_to_local_dir(pretrained_path,
                                     wait_finish=True) as local_path:
            self.model = T5EncoderModel.from_pretrained(
                local_path,
                torch_dtype=getattr(
                    torch,
                    'float' if self.t5_dtype == 'float32' else self.t5_dtype))
        tokenizer_path = cfg.get('TOKENIZER_PATH', None)
        self.length = cfg.get('LENGTH', 77)

        self.use_grad = cfg.get('USE_GRAD', False)
        self.clean = cfg.get('CLEAN', 'whitespace')
        self.added_identifier = cfg.get('ADDED_IDENTIFIER', None)
        if tokenizer_path:
            self.tokenize_kargs = {'return_tensors': 'pt'}
            with FS.get_dir_to_local_dir(tokenizer_path,
                                         wait_finish=True) as local_path:
                if self.added_identifier is not None and isinstance(
                        self.added_identifier, list):
                    self.tokenizer = AutoTokenizer.from_pretrained(local_path)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(local_path)
            if self.length is not None:
                self.tokenize_kargs.update({
                    'padding': 'max_length',
                    'truncation': True,
                    'max_length': self.length
                })
            self.eos_token = self.tokenizer(
                self.tokenizer.eos_token)['input_ids'][0]
        else:
            self.tokenizer = None
            self.tokenize_kargs = {}

        self.use_grad = cfg.get('USE_GRAD', False)
        self.clean = cfg.get('CLEAN', 'whitespace')

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    # encode && encode_text
    def forward(self, tokens, return_mask=False, use_mask=True):
        # tokenization
        embedding_context = nullcontext if self.use_grad else torch.no_grad
        with embedding_context():
            if use_mask:
                x = self.model(tokens.input_ids.to(we.device_id),
                               tokens.attention_mask.to(we.device_id))
            else:
                x = self.model(tokens.input_ids.to(we.device_id))
            x = x.last_hidden_state

            if return_mask:
                return x.detach() + 0.0, tokens.attention_mask.to(we.device_id)
            else:
                return x.detach() + 0.0, None

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        elif self.clean == 'heavy':
            text = heavy_clean(basic_clean(text))
        return text

    def encode(self, text, return_mask=False, use_mask=True):
        if isinstance(text, str):
            text = [text]
        if self.clean:
            text = [self._clean(u) for u in text]
        assert self.tokenizer is not None
        cont, mask = [], []
        with torch.autocast(device_type='cuda',
                            enabled=self.t5_dtype in ('float16', 'bfloat16'),
                            dtype=getattr(torch, self.t5_dtype)):
            for tt in text:
                tokens = self.tokenizer([tt], **self.tokenize_kargs)
                one_cont, one_mask = self(tokens,
                                          return_mask=return_mask,
                                          use_mask=use_mask)
                cont.append(one_cont)
                mask.append(one_mask)
        if return_mask:
            return torch.cat(cont, dim=0), torch.cat(mask, dim=0)
        else:
            return torch.cat(cont, dim=0)

    def encode_list(self, text_list, return_mask=True):
        cont_list = []
        mask_list = []
        for pp in text_list:
            cont, cont_mask = self.encode(pp, return_mask=return_mask)
            cont_list.append(cont)
            mask_list.append(cont_mask)
        if return_mask:
            return cont_list, mask_list
        else:
            return cont_list

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODELS',
                            __class__.__name__,
                            ACETextEmbedder.para_dict,
                            set_name=True)

@EMBEDDERS.register_class()
class ACEHFEmbedder(BaseEmbedder):
    para_dict = {
        "HF_MODEL_CLS": {
            "value": None,
            "description": "huggingface cls in transfomer"
        },
        "MODEL_PATH": {
            "value": None,
            "description": "model folder path"
        },
        "HF_TOKENIZER_CLS": {
            "value": None,
            "description": "huggingface cls in transfomer"
        },

        "TOKENIZER_PATH": {
            "value": None,
            "description": "tokenizer folder path"
        },
        "MAX_LENGTH": {
            "value": 77,
            "description": "max length of input"
        },
        "OUTPUT_KEY": {
            "value": "last_hidden_state",
            "description": "output key"
        },
        "D_TYPE": {
            "value": "float",
            "description": "dtype"
        },
        "BATCH_INFER": {
            "value": False,
            "description": "batch infer"
        }
    }
    para_dict.update(BaseEmbedder.para_dict)
    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        hf_model_cls = cfg.get('HF_MODEL_CLS', None)
        model_path = cfg.get("MODEL_PATH", None)
        hf_tokenizer_cls = cfg.get('HF_TOKENIZER_CLS', None)
        tokenizer_path = cfg.get('TOKENIZER_PATH', None)
        self.max_length = cfg.get('MAX_LENGTH', 77)
        self.output_key = cfg.get("OUTPUT_KEY", "last_hidden_state")
        self.d_type = cfg.get("D_TYPE", "float")
        self.clean = cfg.get("CLEAN", "whitespace")
        self.batch_infer = cfg.get("BATCH_INFER", False)
        self.added_identifier = cfg.get('ADDED_IDENTIFIER', None)
        torch_dtype = getattr(torch, self.d_type)

        assert hf_model_cls is not None and hf_tokenizer_cls is not None
        assert model_path is not None and tokenizer_path is not None
        with FS.get_dir_to_local_dir(tokenizer_path, wait_finish=True) as local_path:
            self.tokenizer = getattr(transformers, hf_tokenizer_cls).from_pretrained(local_path,
                                                                                     max_length = self.max_length,
                                                                                     torch_dtype = torch_dtype,
                                                                                     additional_special_tokens=self.added_identifier)

        with FS.get_dir_to_local_dir(model_path, wait_finish=True) as local_path:
            self.hf_module = getattr(transformers, hf_model_cls).from_pretrained(local_path, torch_dtype = torch_dtype)


        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str], return_mask = False):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        if return_mask:
            return outputs[self.output_key], batch_encoding['attention_mask'].to(self.hf_module.device)
        else:
            return outputs[self.output_key], None

    def encode(self, text, return_mask = False):
        if isinstance(text, str):
            text = [text]
        if self.clean:
            text = [self._clean(u) for u in text]
        if not self.batch_infer:
            cont, mask = [], []
            for tt in text:
                one_cont, one_mask = self([tt], return_mask=return_mask)
                cont.append(one_cont)
                mask.append(one_mask)
            if return_mask:
                return torch.cat(cont, dim=0), torch.cat(mask, dim=0)
            else:
                return torch.cat(cont, dim=0)
        else:
            ret_data = self(text, return_mask = return_mask)
            if return_mask:
                return ret_data
            else:
                return ret_data[0]

    def encode_list(self, text_list, return_mask=True):
        cont_list = []
        mask_list = []
        for pp in text_list:
            cont = self.encode(pp, return_mask=return_mask)
            cont_list.append(cont[0]) if return_mask else cont_list.append(cont)
            mask_list.append(cont[1]) if return_mask else mask_list.append(None)
        if return_mask:
            return cont_list, mask_list
        else:
            return cont_list

    def encode_list_of_list(self, text_list, return_mask=True):
        cont_list = []
        mask_list = []
        for pp in text_list:
            cont = self.encode_list(pp, return_mask=return_mask)
            cont_list.append(cont[0]) if return_mask else cont_list.append(cont)
            mask_list.append(cont[1]) if return_mask else mask_list.append(None)
        if return_mask:
            return cont_list, mask_list
        else:
            return cont_list

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        return text
    @staticmethod
    def get_config_template():
        return dict_to_yaml('EMBEDDER',
                            __class__.__name__,
                            ACEHFEmbedder.para_dict,
                            set_name=True)

@EMBEDDERS.register_class()
class T5ACEPlusClipFluxEmbedder(BaseEmbedder):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    para_dict = {
        'T5_MODEL': {},
        'CLIP_MODEL': {}
    }

    def __init__(self, cfg, logger=None):
        super().__init__(cfg, logger=logger)
        self.t5_model = EMBEDDERS.build(cfg.T5_MODEL, logger=logger)
        self.clip_model = EMBEDDERS.build(cfg.CLIP_MODEL, logger=logger)

    def encode(self, text, return_mask = False):
        t5_embeds = self.t5_model.encode(text, return_mask = return_mask)
        clip_embeds = self.clip_model.encode(text, return_mask = return_mask)
        # change embedding strategy here
        return {
            'context': t5_embeds,
            'y': clip_embeds,
        }

    def encode_list(self, text, return_mask = False):
        t5_embeds = self.t5_model.encode_list(text, return_mask = return_mask)
        clip_embeds = self.clip_model.encode_list(text, return_mask = return_mask)
        # change embedding strategy here
        return {
            'context': t5_embeds,
            'y': clip_embeds,
        }

    def encode_list_of_list(self, text, return_mask = False):
        t5_embeds = self.t5_model.encode_list_of_list(text, return_mask = return_mask)
        clip_embeds = self.clip_model.encode_list_of_list(text, return_mask = return_mask)
        # change embedding strategy here
        return {
            'context': t5_embeds,
            'y': clip_embeds,
        }


    @staticmethod
    def get_config_template():
        return dict_to_yaml('EMBEDDER',
                            __class__.__name__,
                            T5ACEPlusClipFluxEmbedder.para_dict,
                            set_name=True)

@BACKBONES.register_class()
class Flux(BaseModel):
    """
    Transformer backbone Diffusion model with RoPE.
    """
    para_dict = {
        "IN_CHANNELS": {
            "value": 64,
            "description": "model's input channels."
        },
        "OUT_CHANNELS": {
            "value": 64,
            "description": "model's output channels."
        },
        "HIDDEN_SIZE": {
            "value": 1024,
            "description": "model's hidden size."
        },
        "NUM_HEADS": {
            "value": 16,
            "description": "number of heads in the transformer."
        },
        "AXES_DIM": {
            "value": [16, 56, 56],
            "description": "dimensions of the axes of the positional encoding."
        },
        "THETA": {
            "value": 10_000,
            "description": "theta for positional encoding."
        },
        "VEC_IN_DIM": {
            "value": 768,
            "description": "dimension of the vector input."
        },
        "GUIDANCE_EMBED": {
            "value": False,
            "description": "whether to use guidance embedding."
        },
        "CONTEXT_IN_DIM": {
            "value": 4096,
            "description": "dimension of the context input."
        },
        "MLP_RATIO": {
            "value": 4.0,
            "description": "ratio of mlp hidden size to hidden size."
        },
        "QKV_BIAS": {
            "value": True,
            "description": "whether to use bias in qkv projection."
        },
        "DEPTH": {
            "value": 19,
            "description": "number of transformer blocks."
        },
        "DEPTH_SINGLE_BLOCKS": {
            "value": 38,
            "description": "number of transformer blocks in the single stream block."
        },
        "USE_GRAD_CHECKPOINT": {
            "value": False,
            "description": "whether to use gradient checkpointing."
        },
        "ATTN_BACKEND": {
            "value": "pytorch",
            "description": "backend for the transformer blocks, 'pytorch' or 'flash_attn'."
        }
    }
    def __init__(
            self,
            cfg,
            logger = None
    ):
        super().__init__(cfg, logger=logger)
        self.in_channels = cfg.IN_CHANNELS
        self.out_channels = cfg.get("OUT_CHANNELS", self.in_channels)
        hidden_size = cfg.get("HIDDEN_SIZE", 1024)
        num_heads = cfg.get("NUM_HEADS", 16)
        axes_dim = cfg.AXES_DIM
        theta = cfg.THETA
        vec_in_dim = cfg.VEC_IN_DIM
        self.guidance_embed = cfg.GUIDANCE_EMBED
        context_in_dim = cfg.CONTEXT_IN_DIM
        mlp_ratio = cfg.MLP_RATIO
        qkv_bias = cfg.QKV_BIAS
        depth = cfg.DEPTH
        depth_single_blocks = cfg.DEPTH_SINGLE_BLOCKS
        self.use_grad_checkpoint = cfg.get("USE_GRAD_CHECKPOINT", False)
        self.attn_backend = cfg.get("ATTN_BACKEND", "pytorch")
        self.lora_model = cfg.get("DIFFUSERS_LORA_MODEL", None)
        self.swift_lora_model = cfg.get("SWIFT_LORA_MODEL", None)
        self.pretrain_adapter = cfg.get("PRETRAIN_ADAPTER", None)

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim= axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if self.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    backend=self.attn_backend
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=mlp_ratio, backend=self.attn_backend)
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def prepare_input(self, x, context, y, x_shape=None):
        # x.shape [6, 16, 16, 16] target is [6, 16, 768, 1360]
        bs, c, h, w = x.shape
        x = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        x_id = torch.zeros(h // 2, w // 2, 3)
        x_id[..., 1] = x_id[..., 1] + torch.arange(h // 2)[:, None]
        x_id[..., 2] = x_id[..., 2] + torch.arange(w // 2)[None, :]
        x_ids = repeat(x_id, "h w c -> b (h w) c", b=bs)
        txt_ids = torch.zeros(bs, context.shape[1], 3)
        return x, x_ids.to(x), context.to(x), txt_ids.to(x), y.to(x), h, w

    def unpack(self, x: Tensor, height: int, width: int) -> Tensor:
        return rearrange(
            x,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=math.ceil(height/2),
            w=math.ceil(width/2),
            ph=2,
            pw=2,
        )

    def merge_diffuser_lora(self, ori_sd, lora_sd, scale = 1.0):
        key_map = {
            "single_blocks.{}.linear1.weight": {"key_list": [
                ["transformer.single_transformer_blocks.{}.attn.to_q.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.attn.to_q.lora_B.weight"],
                ["transformer.single_transformer_blocks.{}.attn.to_k.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.attn.to_k.lora_B.weight"],
                ["transformer.single_transformer_blocks.{}.attn.to_v.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.attn.to_v.lora_B.weight"],
                ["transformer.single_transformer_blocks.{}.proj_mlp.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.proj_mlp.lora_B.weight"]
            ], "num": 38},
            "single_blocks.{}.modulation.lin.weight": {"key_list": [
                ["transformer.single_transformer_blocks.{}.norm.linear.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.norm.linear.lora_B.weight"],
            ], "num": 38},
            "single_blocks.{}.linear2.weight": {"key_list": [
                ["transformer.single_transformer_blocks.{}.proj_out.lora_A.weight",
                 "transformer.single_transformer_blocks.{}.proj_out.lora_B.weight"],
            ], "num": 38},
            "double_blocks.{}.txt_attn.qkv.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.attn.add_q_proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.add_q_proj.lora_B.weight"],
                ["transformer.transformer_blocks.{}.attn.add_k_proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.add_k_proj.lora_B.weight"],
                ["transformer.transformer_blocks.{}.attn.add_v_proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.add_v_proj.lora_B.weight"],
            ], "num": 19},
            "double_blocks.{}.img_attn.qkv.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.attn.to_q.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_q.lora_B.weight"],
                ["transformer.transformer_blocks.{}.attn.to_k.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_k.lora_B.weight"],
                ["transformer.transformer_blocks.{}.attn.to_v.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_v.lora_B.weight"],
            ], "num": 19},
            "double_blocks.{}.img_attn.proj.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.attn.to_out.0.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_out.0.lora_B.weight"]
            ], "num": 19},
            "double_blocks.{}.txt_attn.proj.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.attn.to_add_out.lora_A.weight",
                 "transformer.transformer_blocks.{}.attn.to_add_out.lora_B.weight"]
            ], "num": 19},
            "double_blocks.{}.img_mlp.0.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.ff.net.0.proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.ff.net.0.proj.lora_B.weight"]
            ], "num": 19},
            "double_blocks.{}.img_mlp.2.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.ff.net.2.lora_A.weight",
                 "transformer.transformer_blocks.{}.ff.net.2.lora_B.weight"]
            ], "num": 19},
            "double_blocks.{}.txt_mlp.0.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.ff_context.net.0.proj.lora_A.weight",
                 "transformer.transformer_blocks.{}.ff_context.net.0.proj.lora_B.weight"]
            ], "num": 19},
            "double_blocks.{}.txt_mlp.2.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.ff_context.net.2.lora_A.weight",
                 "transformer.transformer_blocks.{}.ff_context.net.2.lora_B.weight"]
            ], "num": 19},
            "double_blocks.{}.img_mod.lin.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.norm1.linear.lora_A.weight",
                 "transformer.transformer_blocks.{}.norm1.linear.lora_B.weight"]
            ], "num": 19},
            "double_blocks.{}.txt_mod.lin.weight": {"key_list": [
                ["transformer.transformer_blocks.{}.norm1_context.linear.lora_A.weight",
                 "transformer.transformer_blocks.{}.norm1_context.linear.lora_B.weight"]
            ], "num": 19}
        }
        for k, v in key_map.items():
            key_list = v["key_list"]
            block_num = v["num"]
            for block_id in range(block_num):
                current_weight_list = []
                for k_list in key_list:
                    current_weight = torch.matmul(lora_sd[k_list[0].format(block_id)].permute(1, 0),
                                                  lora_sd[k_list[1].format(block_id)].permute(1, 0)).permute(1, 0)
                    current_weight_list.append(current_weight)
                current_weight = torch.cat(current_weight_list, dim=0)
                ori_sd[k.format(block_id)] += scale*current_weight
        return ori_sd

    def merge_swift_lora(self, ori_sd, lora_sd, scale = 1.0):
        have_lora_keys = {}
        for k, v in lora_sd.items():
            k = k[len("model."):] if k.startswith("model.") else k
            ori_key = k.split("lora")[0] + "weight"
            if ori_key not in ori_sd:
                raise f"{ori_key} should in the original statedict"
            if ori_key not in have_lora_keys:
                have_lora_keys[ori_key] = {}
            if "lora_A" in k:
                have_lora_keys[ori_key]["lora_A"] = v
            elif "lora_B" in k:
                have_lora_keys[ori_key]["lora_B"] = v
            else:
                raise NotImplementedError
        for key, v in have_lora_keys.items():
            current_weight = torch.matmul(v["lora_A"].permute(1, 0), v["lora_B"].permute(1, 0)).permute(1, 0)
            ori_sd[key] += scale * current_weight
        return ori_sd


    def load_pretrained_model(self, pretrained_model):
        if next(self.parameters()).device.type == 'meta':
            map_location = torch.device(we.device_id)
        else:
            map_location = "cpu"
        if self.lora_model is not None:
            map_location = we.device_id
        if pretrained_model is not None:
            with FS.get_from(pretrained_model, wait_finish=True) as local_model:
                if local_model.endswith('safetensors'):
                    from safetensors.torch import load_file as load_safetensors
                    sd = load_safetensors(local_model, device=map_location)
                else:
                    sd = torch.load(local_model, map_location=map_location)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            if "model" in sd:
                sd = sd["model"]["model"]

            if self.lora_model is not None:
                with FS.get_from(self.lora_model, wait_finish=True) as local_model:
                    if local_model.endswith('safetensors'):
                        from safetensors.torch import load_file as load_safetensors
                        lora_sd = load_safetensors(local_model, device=map_location)
                    else:
                        lora_sd = torch.load(local_model, map_location=map_location)
                sd = self.merge_diffuser_lora(sd, lora_sd)
            if self.swift_lora_model is not None:
                with FS.get_from(self.swift_lora_model, wait_finish=True) as local_model:
                    if local_model.endswith('safetensors'):
                        from safetensors.torch import load_file as load_safetensors
                        lora_sd = load_safetensors(local_model, device=map_location)
                    else:
                        lora_sd = torch.load(local_model, map_location=map_location)
                sd = self.merge_swift_lora(sd, lora_sd)

            adapter_ckpt = {}
            if self.pretrain_adapter is not None:
                with FS.get_from(self.pretrain_adapter, wait_finish=True) as local_adapter:
                    if local_model.endswith('safetensors'):
                        from safetensors.torch import load_file as load_safetensors
                        adapter_ckpt = load_safetensors(local_adapter, device=map_location)
                    else:
                        adapter_ckpt = torch.load(local_adapter, map_location=map_location)
            sd.update(adapter_ckpt)


            new_ckpt = OrderedDict()
            for k, v in sd.items():
                if k in ("img_in.weight"):
                    model_p = self.state_dict()[k]
                    if v.shape != model_p.shape:
                        model_p.zero_()
                        model_p[:, :64].copy_(v[:, :64])
                        new_ckpt[k] = torch.nn.parameter.Parameter(model_p)
                    else:
                        new_ckpt[k] = v
                else:
                    new_ckpt[k] = v


            missing, unexpected = self.load_state_dict(new_ckpt, strict=False, assign=True)
            self.logger.info(
                f'Restored from {pretrained_model} with {len(missing)} missing and {len(unexpected)} unexpected keys'
            )
            if len(missing) > 0:
                self.logger.info(f'Missing Keys:\n {missing}')
            if len(unexpected) > 0:
                self.logger.info(f'\nUnexpected Keys:\n {unexpected}')

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        cond: dict = {},
        guidance: Tensor | None = None,
        gc_seg: int = 0
    ) -> Tensor:
        x, x_ids, txt, txt_ids, y, h, w = self.prepare_input(x, cond["context"], cond["y"])
        # running on sequences img
        x = self.img_in(x)
        vec = self.time_in(timestep_embedding(t, 256))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        ids = torch.cat((txt_ids, x_ids), dim=1)
        pe = self.pe_embedder(ids)
        kwargs = dict(
            vec=vec,
            pe=pe,
            txt_length=txt.shape[1],
        )
        x = torch.cat((txt, x), 1)
        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.double_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.double_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.double_blocks:
                x = block(x, **kwargs)

        kwargs = dict(
            vec=vec,
            pe=pe,
        )

        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.single_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.single_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.single_blocks:
                x = block(x, **kwargs)
        x = x[:, txt.shape[1] :, ...]
        x = self.final_layer(x, vec)  # (N, T, patch_size ** 2 * out_channels) 6 64 64
        x = self.unpack(x, h, w)
        return x

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            Flux.para_dict,
                            set_name=True)

@BACKBONES.register_class()
class FluxMR(Flux):
    def prepare_input(self, x, cond):
        if isinstance(cond['context'], list):
            context, y = torch.cat(cond["context"], dim=0).to(x), torch.cat(cond["y"], dim=0).to(x)
        else:
            context, y = cond['context'].to(x), cond['y'].to(x)
        batch_frames, batch_frames_ids = [], []
        for ix, shape in zip(x, cond["x_shapes"]):
            # unpack image from sequence
            ix = ix[:, :shape[0] * shape[1]].view(-1, shape[0], shape[1])
            c, h, w = ix.shape
            ix = rearrange(ix, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
            ix_id = torch.zeros(h // 2, w // 2, 3)
            ix_id[..., 1] = ix_id[..., 1] + torch.arange(h // 2)[:, None]
            ix_id[..., 2] = ix_id[..., 2] + torch.arange(w // 2)[None, :]
            ix_id = rearrange(ix_id, "h w c -> (h w) c")
            batch_frames.append([ix])
            batch_frames_ids.append([ix_id])

        x_list, x_id_list, mask_x_list, x_seq_length = [], [], [], []
        for frames, frame_ids in zip(batch_frames, batch_frames_ids):
            proj_frames = []
            for idx, one_frame in enumerate(frames):
                one_frame = self.img_in(one_frame)
                proj_frames.append(one_frame)
            ix = torch.cat(proj_frames, dim=0)
            if_id = torch.cat(frame_ids, dim=0)
            x_list.append(ix)
            x_id_list.append(if_id)
            mask_x_list.append(torch.ones(ix.shape[0]).to(ix.device, non_blocking=True).bool())
            x_seq_length.append(ix.shape[0])
        x = pad_sequence(tuple(x_list), batch_first=True)
        x_ids = pad_sequence(tuple(x_id_list), batch_first=True).to(x)  # [b,pad_seq,2] pad (0.,0.) at dim2
        mask_x = pad_sequence(tuple(mask_x_list), batch_first=True)

        txt = self.txt_in(context)
        txt_ids = torch.zeros(context.shape[0], context.shape[1], 3).to(x)
        mask_txt = torch.ones(context.shape[0], context.shape[1]).to(x.device, non_blocking=True).bool()

        return x, x_ids, txt, txt_ids, y, mask_x, mask_txt, x_seq_length

    def unpack(self, x: Tensor, cond: dict = None, x_seq_length: list = None) -> Tensor:
        x_list = []
        image_shapes = cond["x_shapes"]
        for u, shape, seq_length in zip(x, image_shapes, x_seq_length):
            height, width = shape
            h, w = math.ceil(height / 2), math.ceil(width / 2)
            u = rearrange(
                u[seq_length-h*w:seq_length, ...],
                "(h w) (c ph pw) -> (h ph w pw) c",
                h=h,
                w=w,
                ph=2,
                pw=2,
            )
            x_list.append(u)
        x = pad_sequence(tuple(x_list), batch_first=True).permute(0, 2, 1)
        return x

    def forward(
            self,
            x: Tensor,
            t: Tensor,
            cond: dict = {},
            guidance: Tensor | None = None,
            gc_seg: int = 0,
            **kwargs
    ) -> Tensor:
        x, x_ids, txt, txt_ids, y, mask_x, mask_txt, seq_length_list = self.prepare_input(x, cond)
        # running on sequences img
        vec = self.time_in(timestep_embedding(t, 256))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        ids = torch.cat((txt_ids, x_ids), dim=1)
        pe = self.pe_embedder(ids)

        mask_aside = torch.cat((mask_txt, mask_x), dim=1)
        mask = mask_aside[:, None, :] * mask_aside[:, :, None]

        kwargs = dict(
            vec=vec,
            pe=pe,
            mask=mask,
            txt_length = txt.shape[1],
        )
        x = torch.cat((txt, x), 1)
        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.double_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.double_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.double_blocks:
                x = block(x, **kwargs)

        kwargs = dict(
            vec=vec,
            pe=pe,
            mask=mask,
        )

        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.single_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.single_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.single_blocks:
                x = block(x, **kwargs)
        x = x[:, txt.shape[1]:, ...]
        x = self.final_layer(x, vec)  # (N, T, patch_size ** 2 * out_channels) 6 64 64
        x = self.unpack(x, cond, seq_length_list)
        return x

    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            FluxEdit.para_dict,
                            set_name=True)
@BACKBONES.register_class()
class FluxEdit(FluxMR):
    def prepare_input(self, x, cond, *args, **kwargs):
        context, y = cond["context"], cond["y"]
        batch_frames, batch_frames_ids, batch_shift = [], [], []

        for ix, shape, is_align in zip(x, cond["x_shapes"], cond['align']):
            # unpack image from sequence
            ix = ix[:, :shape[0] * shape[1]].view(-1, shape[0], shape[1])
            c, h, w = ix.shape
            ix = rearrange(ix, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
            ix_id = torch.zeros(h // 2, w // 2, 3)
            ix_id[..., 1] = ix_id[..., 1] + torch.arange(h // 2)[:, None]
            ix_id[..., 2] = ix_id[..., 2] + torch.arange(w // 2)[None, :]
            batch_shift.append(h // 2) #if is_align < 1 else batch_shift.append(0)
            ix_id = rearrange(ix_id, "h w c -> (h w) c")
            batch_frames.append([ix])
            batch_frames_ids.append([ix_id])
        if 'edit_x' in cond:
            for i, edit in enumerate(cond['edit_x']):
                if edit is None:
                    continue
                for ie in edit:
                    ie = ie.squeeze(0)
                    c, h, w = ie.shape
                    ie = rearrange(ie, "c (h ph) (w pw) -> (h w) (c ph pw)", ph=2, pw=2)
                    ie_id = torch.zeros(h // 2, w // 2, 3)
                    ie_id[..., 1] = ie_id[..., 1] + torch.arange(batch_shift[i], h // 2 + batch_shift[i])[:, None]
                    ie_id[..., 2] = ie_id[..., 2] + torch.arange(w // 2)[None, :]
                    ie_id = rearrange(ie_id, "h w c -> (h w) c")
                    batch_frames[i].append(ie)
                    batch_frames_ids[i].append(ie_id)

        x_list, x_id_list, mask_x_list, x_seq_length = [], [], [], []
        for frames, frame_ids in zip(batch_frames, batch_frames_ids):
            proj_frames = []
            for idx, one_frame in enumerate(frames):
                one_frame = self.img_in(one_frame)
                proj_frames.append(one_frame)
            ix = torch.cat(proj_frames, dim=0)
            if_id = torch.cat(frame_ids, dim=0)
            x_list.append(ix)
            x_id_list.append(if_id)
            mask_x_list.append(torch.ones(ix.shape[0]).to(ix.device, non_blocking=True).bool())
            x_seq_length.append(ix.shape[0])
        x = pad_sequence(tuple(x_list), batch_first=True)
        x_ids = pad_sequence(tuple(x_id_list), batch_first=True).to(x)  # [b,pad_seq,2] pad (0.,0.) at dim2
        mask_x = pad_sequence(tuple(mask_x_list), batch_first=True)

        txt_list, mask_txt_list, y_list = [], [], []
        for sample_id, (ctx, yy) in enumerate(zip(context, y)):
            ctx_batch = []
            for frame_id, one_ctx in enumerate(ctx):
                one_ctx = self.txt_in(one_ctx.to(x))
                ctx_batch.append(one_ctx)
            txt_list.append(torch.cat(ctx_batch, dim=0))
            mask_txt_list.append(torch.ones(txt_list[-1].shape[0]).to(ctx.device, non_blocking=True).bool())
            y_list.append(yy.mean(dim = 0, keepdim=True))
        txt = pad_sequence(tuple(txt_list), batch_first=True)
        txt_ids = torch.zeros(txt.shape[0], txt.shape[1], 3).to(x)
        mask_txt = pad_sequence(tuple(mask_txt_list), batch_first=True)
        y = torch.cat(y_list, dim=0)
        return x, x_ids, txt, txt_ids, y, mask_x, mask_txt, x_seq_length

    def unpack(self, x: Tensor, cond: dict = None, x_seq_length: list = None) -> Tensor:
        x_list = []
        image_shapes = cond["x_shapes"]
        for u, shape, seq_length in zip(x, image_shapes, x_seq_length):
            height, width = shape
            h, w = math.ceil(height / 2), math.ceil(width / 2)
            u = rearrange(
                u[:h*w, ...],
                "(h w) (c ph pw) -> (h ph w pw) c",
                h=h,
                w=w,
                ph=2,
                pw=2,
            )
            x_list.append(u)
        x = pad_sequence(tuple(x_list), batch_first=True).permute(0, 2, 1)
        return x

    def forward(
            self,
            x: Tensor,
            t: Tensor,
            cond: dict = {},
            guidance: Tensor | None = None,
            gc_seg: int = 0,
            text_position_embeddings = None
    ) -> Tensor:
        x, x_ids, txt, txt_ids, y, mask_x, mask_txt, seq_length_list = self.prepare_input(x, cond, text_position_embeddings)
        # running on sequences img
        vec = self.time_in(timestep_embedding(t, 256))
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        ids = torch.cat((txt_ids, x_ids), dim=1)
        pe = self.pe_embedder(ids)

        mask_aside = torch.cat((mask_txt, mask_x), dim=1)
        mask = mask_aside[:, None, :] * mask_aside[:, :, None]

        kwargs = dict(
            vec=vec,
            pe=pe,
            mask=mask,
            txt_length = txt.shape[1],
        )
        x = torch.cat((txt, x), 1)

        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.double_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.double_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.double_blocks:
                x = block(x, **kwargs)

        kwargs = dict(
            vec=vec,
            pe=pe,
            mask=mask,
        )

        if self.use_grad_checkpoint and gc_seg >= 0:
            x = checkpoint_sequential(
                functions=[partial(block, **kwargs) for block in self.single_blocks],
                segments=gc_seg if gc_seg > 0 else len(self.single_blocks),
                input=x,
                use_reentrant=False
            )
        else:
            for block in self.single_blocks:
                x = block(x, **kwargs)
        x = x[:, txt.shape[1]:, ...]
        x = self.final_layer(x, vec)  # (N, T, patch_size ** 2 * out_channels) 6 64 64
        x = self.unpack(x, cond, seq_length_list)
        return x
    @staticmethod
    def get_config_template():
        return dict_to_yaml('MODEL',
                            __class__.__name__,
                            FluxEdit.para_dict,
                            set_name=True)