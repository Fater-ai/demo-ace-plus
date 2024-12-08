# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as T
from scepter.modules.model.registry import DIFFUSIONS,BACKBONES
from scepter.modules.model.utils.basic_utils import check_list_of_list
from scepter.modules.model.utils.basic_utils import \
    pack_imagelist_into_tensor_v2 as pack_imagelist_into_tensor
from scepter.modules.model.utils.basic_utils import (
    to_device, unpack_tensor_into_imagelist)
from scepter.modules.utils.distribute import we
from scepter.modules.utils.logger import get_logger

from scepter.modules.inference.diffusion_inference import DiffusionInference, get_model


def process_edit_image(images,
                       masks,
                       tasks,
                       max_seq_len=1024,
                       max_aspect_ratio=4,
                       d=16,
                       **kwargs):

    if not isinstance(images, list):
        images = [images]
    if not isinstance(masks, list):
        masks = [masks]
    if not isinstance(tasks, list):
        tasks = [tasks]

    img_tensors = []
    mask_tensors = []
    for img, mask, task in zip(images, masks, tasks):
        if mask is None or mask == '':
            mask = Image.new('L', img.size, 0)
        W, H = img.size
        if H / W > max_aspect_ratio:
            img = TF.center_crop(img, [int(max_aspect_ratio * W), W])
            mask = TF.center_crop(mask, [int(max_aspect_ratio * W), W])
        elif W / H > max_aspect_ratio:
            img = TF.center_crop(img, [H, int(max_aspect_ratio * H)])
            mask = TF.center_crop(mask, [H, int(max_aspect_ratio * H)])

        H, W = img.height, img.width
        scale = min(1.0, math.sqrt(max_seq_len / ((H / d) * (W / d))))
        rH = int(H * scale) // d * d  # ensure divisible by self.d
        rW = int(W * scale) // d * d

        img = TF.resize(img, (rH, rW),
                        interpolation=TF.InterpolationMode.BICUBIC)
        mask = TF.resize(mask, (rH, rW),
                         interpolation=TF.InterpolationMode.NEAREST_EXACT)

        mask = np.asarray(mask)
        mask = np.where(mask > 128, 1, 0)
        mask = mask.astype(
            np.float32) if np.any(mask) else np.ones_like(mask).astype(
                np.float32)

        img_tensor = TF.to_tensor(img).to(we.device_id)
        img_tensor = TF.normalize(img_tensor,
                                  mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
        mask_tensor = TF.to_tensor(mask).to(we.device_id)
        if task in ['inpainting', 'Try On', 'Inpainting']:
            mask_indicator = mask_tensor.repeat(3, 1, 1)
            img_tensor[mask_indicator == 1] = -1.0
        img_tensors.append(img_tensor)
        mask_tensors.append(mask_tensor)
    return img_tensors, mask_tensors

class TextEmbedding(nn.Module):
    def __init__(self, embedding_shape):
        super().__init__()
        self.pos = nn.Parameter(data=torch.zeros(embedding_shape))

class ACEInference(DiffusionInference):
    def __init__(self, logger=None):
        if logger is None:
            logger = get_logger(name='scepter')
        self.logger = logger
        self.loaded_model = {}
        self.loaded_model_name = [
            'diffusion_model', 'first_stage_model', 'cond_stage_model', 'ref_cond_stage_model'
        ]

    def init_from_cfg(self, cfg):
        self.name = cfg.NAME
        self.is_default = cfg.get('IS_DEFAULT', False)
        self.use_dynamic_model = cfg.get('USE_DYNAMIC_MODEL', True)
        module_paras = self.load_default(cfg.get('DEFAULT_PARAS', None))
        assert cfg.have('MODEL')
        self.size_factor = cfg.get('SIZE_FACTOR', 8)
        self.diffusion_model = self.infer_model(
            cfg.MODEL.DIFFUSION_MODEL, module_paras.get(
                'DIFFUSION_MODEL',
                None)) if cfg.MODEL.have('DIFFUSION_MODEL') else None
        self.first_stage_model = self.infer_model(
            cfg.MODEL.FIRST_STAGE_MODEL,
            module_paras.get(
                'FIRST_STAGE_MODEL',
                None)) if cfg.MODEL.have('FIRST_STAGE_MODEL') else None
        self.cond_stage_model = self.infer_model(
            cfg.MODEL.COND_STAGE_MODEL,
            module_paras.get(
                'COND_STAGE_MODEL',
                None)) if cfg.MODEL.have('COND_STAGE_MODEL') else None

        self.ref_cond_stage_model = self.infer_model(
            cfg.MODEL.REF_COND_STAGE_MODEL,
            module_paras.get(
                'REF_COND_STAGE_MODEL',
                None)) if cfg.MODEL.have('REF_COND_STAGE_MODEL') else None

        self.diffusion = DIFFUSIONS.build(cfg.MODEL.DIFFUSION,
                                          logger=self.logger)
        self.interpolate_func = lambda x: (F.interpolate(
            x.unsqueeze(0),
            scale_factor=1 / self.size_factor,
            mode='nearest-exact') if x is not None else None)

        self.max_seq_length = cfg.get("MAX_SEQ_LENGTH", 4096)
        self.src_max_seq_length = cfg.get("SRC_MAX_SEQ_LENGTH", 1024)
        self.image_token = cfg.MODEL.get("IMAGE_TOKEN", "<img>")

        self.text_indentifers = cfg.MODEL.get('TEXT_IDENTIFIER', [])
        self.use_text_pos_embeddings = cfg.MODEL.get('USE_TEXT_POS_EMBEDDINGS',
                                                     False)
        if self.use_text_pos_embeddings:
            self.text_position_embeddings = TextEmbedding(
                (10, 4096)).eval().requires_grad_(False).to(we.device_id)
        else:
            self.text_position_embeddings = None

        if not self.use_dynamic_model:
            self.dynamic_load(self.first_stage_model, 'first_stage_model')
            self.dynamic_load(self.cond_stage_model, 'cond_stage_model')
            if self.ref_cond_stage_model is not None: self.dynamic_load(self.ref_cond_stage_model, 'ref_cond_stage_model')
            # self.dynamic_load(self.diffusion_model, 'diffusion_model')
            # self.diffusion_model["model"].to(torch.bfloat16)
            with torch.device("meta"):
                pretrained_model = self.diffusion_model['cfg'].PRETRAINED_MODEL
                self.diffusion_model['cfg'].PRETRAINED_MODEL = None
                self.diffusion_model['model'] = BACKBONES.build(self.diffusion_model['cfg'], logger=self.logger).eval()
                # self.dynamic_load(self.diffusion_model, 'diffusion_model')
                self.diffusion_model['model'].load_pretrained_model(pretrained_model)
                self.diffusion_model['device'] = we.device_id

    def upscale_resize(self, image, interpolation=T.InterpolationMode.BILINEAR):
        c, H, W = image.shape
        scale = max(1.0, math.sqrt(self.max_seq_length / ((H / 16) * (W / 16))))
        rH = int(H * scale) // 16 * 16  # ensure divisible by self.d
        rW = int(W * scale) // 16 * 16
        image = T.Resize((rH, rW), interpolation=interpolation, antialias=True)(image)
        return image


    @torch.no_grad()
    def encode_first_stage(self, x, **kwargs):
        _, dtype = self.get_function_info(self.first_stage_model, 'encode')
        with torch.autocast('cuda',
                            enabled=dtype in ('float16', 'bfloat16'),
                            dtype=getattr(torch, dtype)):
            def run_one_image(u):
                zu = get_model(self.first_stage_model).encode(u)
                if isinstance(zu, (tuple, list)):
                    zu = zu[0]
                return zu

            z = [run_one_image(u.unsqueeze(0) if u.dim() == 3 else u) for u in x]
            return z


    @torch.no_grad()
    def decode_first_stage(self, z):
        _, dtype = self.get_function_info(self.first_stage_model, 'decode')
        with torch.autocast('cuda',
                            enabled=dtype in ('float16', 'bfloat16'),
                            dtype=getattr(torch, dtype)):
            return [get_model(self.first_stage_model).decode(zu) for zu in z]

    def noise_sample(self, num_samples, h, w, seed, device = None, dtype = torch.bfloat16):
        noise = torch.randn(
            num_samples,
            16,
            # allow for packing
            2 * math.ceil(h / 16),
            2 * math.ceil(w / 16),
            device=device,
            dtype=dtype,
            generator=torch.Generator(device=device).manual_seed(seed),
        )
        return noise

    # def preprocess_prompt(self, prompt):
    #     prompt_ = [[pp] if isinstance(pp, str) else pp for pp in prompt]
    #     for pp_id, pp in enumerate(prompt_):
    #         prompt_[pp_id] = [""] + pp
    #         for p_id, p in enumerate(prompt_[pp_id]):
    #             prompt_[pp_id][p_id] = self.image_token + self.text_indentifers[p_id] + " " + p
    #         prompt_[pp_id] = [f";".join(prompt_[pp_id])]
    #     return prompt_

    @torch.no_grad()
    def __call__(self,
                 image=None,
                 mask=None,
                 prompt='',
                 task=None,
                 negative_prompt='',
                 output_height=1024,
                 output_width=1024,
                 sampler='flow_euler',
                 sample_steps=20,
                 guide_scale=3.5,
                 seed=-1,
                 history_io=None,
                 tar_index=0,
                 align=0,
                 **kwargs):
        input_image, input_mask = image, mask
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        if input_image is not None:
            # assert isinstance(input_image, list) and isinstance(input_mask, list)
            if task is None:
                task = [''] * len(input_image)
            if not isinstance(prompt, list):
                prompt = [prompt] * len(input_image)
            prompt = [
                pp.replace('{image}', f'{{image{i}}}') if i > 0 else pp
                for i, pp in enumerate(prompt)
            ]
            edit_image, edit_image_mask = process_edit_image(
                input_image, input_mask, task, max_seq_len=self.src_max_seq_length)
            image, image_mask = self.upscale_resize(edit_image[tar_index]), self.upscale_resize(edit_image_mask[
               tar_index])
            # edit_image, edit_image_mask = [[self.upscale_resize(i) for i in edit_image]], [[self.upscale_resize(i) for i in edit_image_mask]]
            # image, image_mask = edit_image[tar_index], edit_image_mask[tar_index]
            edit_image, edit_image_mask = [edit_image], [edit_image_mask]
        else:
            edit_image = edit_image_mask = [[]]
            image = torch.zeros(
                size=[3, int(output_height),
                      int(output_width)])
            image_mask = torch.ones(
                size=[1, int(output_height),
                      int(output_width)])
            if not isinstance(prompt, list):
                prompt = [prompt]

        image, image_mask, prompt = [image], [image_mask], [prompt],
        align = [align for p in prompt] if isinstance(align, int) else align

        assert check_list_of_list(prompt) and check_list_of_list(
            edit_image) and check_list_of_list(edit_image_mask)
        # negative prompt is not used
        image = to_device(image)
        ctx = {}
        # Get Noise Shape
        self.dynamic_load(self.first_stage_model, 'first_stage_model')
        x = self.encode_first_stage(image)
        self.dynamic_unload(self.first_stage_model,
                            'first_stage_model',
                            skip_loaded=not self.use_dynamic_model)

        g = torch.Generator(device=we.device_id).manual_seed(seed)

        noise = [
            torch.randn((1, 16, i.shape[2], i.shape[3]), device=we.device_id, dtype=torch.bfloat16).normal_(generator=g)
            for i in x
        ]
        noise, x_shapes = pack_imagelist_into_tensor(noise)
        ctx['x_shapes'] = x_shapes
        ctx['align'] = align

        image_mask = to_device(image_mask, strict=False)
        cond_mask = [self.interpolate_func(i) for i in image_mask
                     ] if image_mask is not None else [None] * len(image)
        ctx['x_mask'] = cond_mask
        # Encode Prompt
        instruction_prompt = [[pp[-1]] if "{image}" in pp[-1] else ["{image} " + pp[-1]] for pp in prompt]
        self.dynamic_load(self.cond_stage_model, 'cond_stage_model')
        function_name, dtype = self.get_function_info(self.cond_stage_model)
        cont = getattr(get_model(self.cond_stage_model), function_name)(instruction_prompt)
        cont["context"] = [ct[-1] for ct in cont["context"]]
        cont["y"] = [ct[-1] for ct in cont["y"]]
        self.dynamic_unload(self.cond_stage_model,
                            'cond_stage_model',
                            skip_loaded=not self.use_dynamic_model)
        ctx.update(cont)

        # Encode Edit Images
        self.dynamic_load(self.first_stage_model, 'first_stage_model')
        edit_image = [to_device(i, strict=False) for i in edit_image]
        edit_image_mask = [to_device(i, strict=False) for i in edit_image_mask]
        e_img, e_mask = [], []
        for u, m in zip(edit_image, edit_image_mask):
            if u is None:
                continue
            if m is None:
                m = [None] * len(u)
            e_img.append(self.encode_first_stage(u, **kwargs))
            e_mask.append([self.interpolate_func(i) for i in m])
        self.dynamic_unload(self.first_stage_model,
                            'first_stage_model',
                            skip_loaded=not self.use_dynamic_model)
        ctx['edit_x'] = e_img
        ctx['edit_mask'] = e_mask
        # Encode Ref Images
        if guide_scale is not None:
            guide_scale = torch.full((noise.shape[0],), guide_scale, device=noise.device, dtype=noise.dtype)
        else:
            guide_scale = None

        # Diffusion Process
        self.dynamic_load(self.diffusion_model, 'diffusion_model')
        function_name, dtype = self.get_function_info(self.diffusion_model)
        with torch.autocast('cuda',
                            enabled=dtype in ('float16', 'bfloat16'),
                            dtype=getattr(torch, dtype)):
            latent = self.diffusion.sample(
                noise=noise,
                sampler=sampler,
                model=get_model(self.diffusion_model),
                model_kwargs={
                    "cond": ctx, "guidance": guide_scale, "gc_seg": -1
                },
                steps=sample_steps,
                show_progress=True,
                guide_scale=guide_scale,
                return_intermediate=None,
                reverse_scale=-1,
                **kwargs).float()
        if self.use_dynamic_model: self.dynamic_unload(self.diffusion_model,
                            'diffusion_model',
                            skip_loaded=not self.use_dynamic_model)

        # Decode to Pixel Space
        self.dynamic_load(self.first_stage_model, 'first_stage_model')
        samples = unpack_tensor_into_imagelist(latent, x_shapes)
        x_samples = self.decode_first_stage(samples)
        self.dynamic_unload(self.first_stage_model,
                            'first_stage_model',
                            skip_loaded=not self.use_dynamic_model)
        x_samples = [x.squeeze(0) for x in x_samples]

        imgs = [
            torch.clamp((x_i.float() + 1.0) / 2.0,
                        min=0.0,
                        max=1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            for x_i in x_samples
        ]
        imgs = [Image.fromarray((img * 255).astype(np.uint8)) for img in imgs]
        return imgs
