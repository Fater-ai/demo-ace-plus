# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import warnings
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.utils.dlpack
from scepter.modules.model.embedder.base_embedder import BaseEmbedder
from scepter.modules.model.registry import EMBEDDERS
from scepter.modules.model.tokenizer.tokenizer_component import (
    basic_clean, canonicalize, heavy_clean, whitespace_clean)
from scepter.modules.utils.config import dict_to_yaml
from scepter.modules.utils.distribute import we
from scepter.modules.utils.file_system import FS

try:
    from transformers import AutoTokenizer, T5EncoderModel
except Exception as e:
    warnings.warn(
        f'Import transformers error, please deal with this problem: {e}')


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