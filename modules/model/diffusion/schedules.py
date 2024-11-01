# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from scepter.modules.model.registry import NOISE_SCHEDULERS
from scepter.modules.model.diffusion.schedules import BaseNoiseScheduler


@NOISE_SCHEDULERS.register_class()
class LinearScheduler(BaseNoiseScheduler):
    para_dict = {}

    def init_params(self):
        super().init_params()
        self.beta_min = self.cfg.get('BETA_MIN', 0.00085)
        self.beta_max = self.cfg.get('BETA_MAX', 0.012)

    def betas_to_sigmas(self, betas):
        return torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))

    def get_schedule(self):
        betas = torch.linspace(self.beta_min,
                               self.beta_max,
                               self.num_timesteps,
                               dtype=torch.float32)
        sigmas = self.betas_to_sigmas(betas)
        self._sigmas = sigmas
        self._betas = betas
        self._alphas = torch.sqrt(1 - sigmas**2)
        self._timesteps = torch.arange(len(sigmas), dtype=torch.float32)