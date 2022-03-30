# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from losses.pcp import PerceptualLoss

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class TwoStageLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, truncation_psi=1, pcp_ratio=1.0):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.truncation_psi = truncation_psi
        self.pcp = PerceptualLoss(layer_weights=dict(conv4_4=1/4, conv5_4=1/2)).to(device)
        self.pcp_ratio = pcp_ratio

    def run_G(self, img_in, mask_in, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c, truncation_psi=self.truncation_psi)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, truncation_psi=self.truncation_psi, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img, img_stg1 = self.G_synthesis(img_in, mask_in, ws, return_stg1=True)
        return img, ws, img_stg1

    def run_D(self, img, mask, img_stg1, c, sync):
        # if self.augment_pipe is not None:
        #     # img = self.augment_pipe(img)
        #     # !!!!! have to remove the color transform
        #     tmp_img = torch.cat([img, mask], dim=1)
        #     tmp_img = self.augment_pipe(tmp_img)
        #     img, mask = torch.split(tmp_img, [3, 1])
        with misc.ddp_sync(self.D, sync):
            logits, logits_stg1 = self.D(img, mask, img_stg1, c)
        return logits, logits_stg1

    def accumulate_gradients(self, phase, real_img, mask, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, gen_img_stg1 = self.run_G(real_img, mask, gen_z, gen_c, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits, gen_logits_stg1 = self.run_D(gen_img, mask, gen_img_stg1, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/scores/fake_s1', gen_logits_stg1)
                training_stats.report('Loss/signs/fake_s1', gen_logits_stg1.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
                loss_Gmain_stg1 = torch.nn.functional.softplus(-gen_logits_stg1)
                training_stats.report('Loss/G/loss_s1', loss_Gmain_stg1)
                # just for showing
                l1_loss = torch.mean(torch.abs(gen_img - real_img))
                training_stats.report('Loss/G/l1_loss', l1_loss)
                pcp_loss, _ = self.pcp(gen_img, real_img)
                training_stats.report('Loss/G/pcp_loss', pcp_loss)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain_all = loss_Gmain + loss_Gmain_stg1 + pcp_loss * self.pcp_ratio
                loss_Gmain_all.mean().mul(gain).backward()

        # # Gpl: Apply path length regularization.
        # if do_Gpl:
        #     with torch.autograd.profiler.record_function('Gpl_forward'):
        #         batch_size = gen_z.shape[0] // self.pl_batch_shrink
        #         gen_img, gen_ws = self.run_G(real_img[:batch_size], mask[:batch_size], gen_z[:batch_size], gen_c[:batch_size], sync=sync)
        #         pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
        #         with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
        #             pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
        #         pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        #         pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        #         self.pl_mean.copy_(pl_mean.detach())
        #         pl_penalty = (pl_lengths - pl_mean).square()
        #         training_stats.report('Loss/pl_penalty', pl_penalty)
        #         loss_Gpl = pl_penalty * self.pl_weight
        #         training_stats.report('Loss/G/reg', loss_Gpl)
        #     with torch.autograd.profiler.record_function('Gpl_backward'):
        #         (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        loss_Dgen_stg1 = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws, gen_img_stg1 = self.run_G(real_img, mask, gen_z, gen_c, sync=False)
                gen_logits, gen_logits_stg1 = self.run_D(gen_img, mask, gen_img_stg1, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
                training_stats.report('Loss/scores/fake_s1', gen_logits_stg1)
                training_stats.report('Loss/signs/fake_s1', gen_logits_stg1.sign())
                loss_Dgen_stg1 = torch.nn.functional.softplus(gen_logits_stg1)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen_all = loss_Dgen + loss_Dgen_stg1
                loss_Dgen_all.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                mask_tmp = mask.detach().requires_grad_(do_Dr1)
                real_img_tmp_stg1 = real_img.detach().requires_grad_(do_Dr1)
                real_logits, real_logits_stg1 = self.run_D(real_img_tmp, mask_tmp, real_img_tmp_stg1, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/scores/real_s1', real_logits_stg1)
                training_stats.report('Loss/signs/real_s1', real_logits_stg1.sign())

                loss_Dreal = 0
                loss_Dreal_stg1 = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    loss_Dreal_stg1 = torch.nn.functional.softplus(-real_logits_stg1)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                    training_stats.report('Loss/D/loss_s1', loss_Dgen_stg1 + loss_Dreal_stg1)

                loss_Dr1 = 0
                loss_Dr1_stg1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                        r1_grads_stg1 = torch.autograd.grad(outputs=[real_logits_stg1.sum()], inputs=[real_img_tmp_stg1], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

                    r1_penalty_stg1 = r1_grads_stg1.square().sum([1, 2, 3])
                    loss_Dr1_stg1 = r1_penalty_stg1 * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty_s1', r1_penalty_stg1)
                    training_stats.report('Loss/D/reg_s1', loss_Dr1_stg1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                ((real_logits + real_logits_stg1) * 0 + loss_Dreal + loss_Dreal_stg1 + loss_Dr1 + loss_Dr1_stg1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
