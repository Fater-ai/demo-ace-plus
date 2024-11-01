# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from scepter.modules.model.registry import DIFFUSION_SAMPLERS
from scepter.modules.model.diffusion.samplers import BaseDiffusionSampler
from scepter.modules.model.diffusion.util import _i

def _i(tensor, t, x):
    """
    Index tensor using t and format the output according to x.
    """
    shape = (x.size(0), ) + (1, ) * (x.ndim - 1)
    if isinstance(t, torch.Tensor):
        t = t.to(tensor.device)
    return tensor[t].view(shape).to(x.device)


@DIFFUSION_SAMPLERS.register_class('ddim')
class DDIMSampler(BaseDiffusionSampler):
    def init_params(self):
        super().init_params()
        self.eta = self.cfg.get('ETA', 0.)
        self.discretization_type = self.cfg.get('DISCRETIZATION_TYPE',
                                                'trailing')

    def preprare_sampler(self,
                         noise,
                         steps=20,
                         scheduler_ins=None,
                         prediction_type='',
                         sigmas=None,
                         betas=None,
                         alphas=None,
                         callback_fn=None,
                         **kwargs):
        output = super().preprare_sampler(noise, steps, scheduler_ins,
                                          prediction_type, sigmas, betas,
                                          alphas, callback_fn, **kwargs)
        sigmas = output.sigmas
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        sigmas_vp = (sigmas**2 / (1 + sigmas**2))**0.5
        sigmas_vp[sigmas == float('inf')] = 1.
        output.add_custom_field('sigmas_vp', sigmas_vp)
        return output

    def step(self, sampler_output):
        x_t = sampler_output.x_t
        step = sampler_output.step
        t = sampler_output.ts[step]
        sigmas_vp = sampler_output.sigmas_vp.to(x_t.device)
        alpha_init = _i(sampler_output.alphas_init, step, x_t[:1])
        sigma_init = _i(sampler_output.sigmas_init, step, x_t[:1])

        x = sampler_output.callback_fn(x_t, t, sigma_init, alpha_init)
        noise_factor = self.eta * (sigmas_vp[step + 1]**2 /
                                   sigmas_vp[step]**2 *
                                   (1 - (1 - sigmas_vp[step]**2) /
                                    (1 - sigmas_vp[step + 1]**2)))
        d = (x_t - (1 - sigmas_vp[step]**2)**0.5 * x) / sigmas_vp[step]
        x = (1 - sigmas_vp[step + 1] ** 2) ** 0.5 * x + \
            (sigmas_vp[step + 1] ** 2 - noise_factor ** 2) ** 0.5 * d
        sampler_output.x_0 = x
        if sigmas_vp[step + 1] > 0:
            x += noise_factor * torch.randn_like(x)
        sampler_output.x_t = x
        sampler_output.step += 1
        sampler_output.msg = f'step {step}'
        return sampler_output