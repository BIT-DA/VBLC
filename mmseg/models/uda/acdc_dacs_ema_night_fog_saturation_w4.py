import os
import random
from copy import deepcopy

import matplotlib.pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import torch
import numpy as np

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module

from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks, get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.models.utils.night_fog_filter import night_fog_filter


@UDA.register_module()
class ACDCDACSEMANightFogSaturationW4(UDADecorator):

    def __init__(self, **cfg):
        super(ACDCDACSEMANightFogSaturationW4, self).__init__(**cfg)
        self.local_iter = 0
        self.debug_img_interval = cfg.get('debug_img_interval', None)
        self.ignore_index = 255
        self.pseudo_threshold = cfg.get('pseudo_threshold', 0.968)
        self.alpha = cfg['alpha']

        # dacs transform
        self.blur = cfg['blur']
        self.color_jitter = cfg['color_jitter']
        self.color_jitter_s = cfg.get('color_jitter_strength', 0.2)
        self.color_jitter_p = cfg.get('color_jitter_probability', 0.2)

        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def forward_train(self, img, img_metas, gt_semantic_seg, return_feat=False):

        log_vars = {}
        batch_size = img[0][0].shape[0]
        dev = img[0][0].device
        means, stds = get_mean_std(img_metas[0][0], dev)

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        src_img, tgt_img = img[0][0], img[1][0]
        src_img_metas, tgt_img_metas = img_metas[0][0], img_metas[1][0]
        src_gt_semantic_seg, tgt_gt_semantic_seg = gt_semantic_seg[0][0], gt_semantic_seg[1][0]

        # illumination boost image
        night_map = []
        for meta in tgt_img_metas:
            if 'night' in meta['filename']:
                night_map.append(1)
            else:
                night_map.append(0)
        tgt_ib_img = night_fog_filter(tgt_img, means, stds, night_map, mode='hsv-s-w4')

        # train main model with source
        src_losses = self.get_model().forward_train(
            src_img, src_img_metas, src_gt_semantic_seg, return_feat=False)
        src_loss, src_log_vars = self._parse_losses(src_losses)
        log_vars.update(add_prefix(src_log_vars, f'src'))
        src_loss.backward()

        # generate target pseudo label from aux model
        tgt_logits = self.get_ema_model().encode_decode(tgt_ib_img, tgt_img_metas)
        tgt_softmax = torch.softmax(tgt_logits.detach(), dim=1)
        tgt_prob, pseudo_label = torch.max(tgt_softmax, dim=1)
        tgt_pseudo_mask = tgt_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(tgt_pseudo_mask).item() / ps_size

        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_label.shape, device=dev)
        gt_pixel_weight = torch.ones(pseudo_weight.shape, device=dev)

        # prepare for dacs transforms
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1) if self.color_jitter else 0,
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # dacs mixed target
        mix_masks = get_class_masks(src_gt_semantic_seg)

        mixed_img, mixed_lbl, mixed_ib_img = [None] * batch_size, [None] * batch_size, [None] * batch_size
        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((src_img[i], tgt_img[i])),
                target=torch.stack((src_gt_semantic_seg[i][0], pseudo_label[i])))
            mixed_ib_img[i], pseudo_weight[i] = strong_transform(
                strong_parameters,
                data=torch.stack((src_img[i], tgt_ib_img[i])),
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))

        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        mixed_ib_img = torch.cat(mixed_ib_img)

        # train main model with target
        mix_losses = self.get_model().forward_train(
            mixed_img, tgt_img_metas, mixed_lbl, pseudo_weight, return_feat=False)
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(add_prefix(mix_log_vars, 'mix'))
        mix_loss.backward()

        # train main model with target ib
        mix_ib_losses = self.get_model().forward_train(
            mixed_ib_img, tgt_img_metas, mixed_lbl, pseudo_weight, return_feat=False)
        mix_ib_loss, mix_ib_log_vars = self._parse_losses(mix_ib_losses)
        log_vars.update(add_prefix(mix_ib_log_vars, 'mix_ib'))
        mix_ib_loss.backward()

        # visualize
        if self.debug_img_interval is not None and self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'visualize_meta')
            os.makedirs(out_dir, exist_ok=True)
            vis_src_img = torch.clamp(denorm(src_img, means, stds), 0, 1)
            vis_tgt_img = torch.clamp(denorm(tgt_img, means, stds), 0, 1)
            vis_mix_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            vis_tgt_ib_img = torch.clamp(denorm(tgt_ib_img, means, stds), 0, 1)
            vis_mix_ib_img = torch.clamp(denorm(mixed_ib_img, means, stds), 0, 1)
            with torch.no_grad():
                # source pseudo label
                src_logits = self.get_model().encode_decode(src_img, src_img_metas)
                src_softmax = torch.softmax(src_logits.detach(), dim=1)
                _, src_pseudo_label = torch.max(src_softmax, dim=1)
                src_pseudo_label = src_pseudo_label.unsqueeze(1)
                # source ema label
                src_logits = self.get_ema_model().encode_decode(src_img, src_img_metas)
                src_softmax = torch.softmax(src_logits.detach(), dim=1)
                _, src_ema_label = torch.max(src_softmax, dim=1)
                src_ema_label = src_ema_label.unsqueeze(1)
                # target pseudo label
                tgt_logits = self.get_model().encode_decode(tgt_img, tgt_img_metas)
                tgt_softmax = torch.softmax(tgt_logits.detach(), dim=1)
                _, tgt_pseudo_label = torch.max(tgt_softmax, dim=1)
                tgt_pseudo_label = tgt_pseudo_label.unsqueeze(1)
                # target ib label
                tgt_logits = self.get_model().encode_decode(tgt_ib_img, tgt_img_metas)
                tgt_softmax = torch.softmax(tgt_logits.detach(), dim=1)
                _, tgt_ib_label = torch.max(tgt_softmax, dim=1)
                tgt_ib_label = tgt_ib_label.unsqueeze(1)
                # target ema label
                tgt_logits = self.get_ema_model().encode_decode(tgt_ib_img, tgt_img_metas)
                tgt_softmax = torch.softmax(tgt_logits.detach(), dim=1)
                _, tgt_ema_ib_label = torch.max(tgt_softmax, dim=1)
                tgt_ema_ib_label = tgt_ema_ib_label.unsqueeze(1)
                # mixed label pred
                mix_logits = self.get_model().encode_decode(mixed_img, tgt_img_metas)
                mix_softmax = torch.softmax(mix_logits.detach(), dim=1)
                _, mix_label_test = torch.max(mix_softmax, dim=1)
                mix_label_test = mix_label_test.unsqueeze(1)
                # mixed ib label pred
                mix_logits = self.get_model().encode_decode(mixed_ib_img, tgt_img_metas)
                mix_softmax = torch.softmax(mix_logits.detach(), dim=1)
                _, mix_ib_label_test = torch.max(mix_softmax, dim=1)
                mix_ib_label_test = mix_ib_label_test.unsqueeze(1)

            for j in range(batch_size):
                rows, cols = 3, 6
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                # source visualization
                subplotimg(axs[0][0], vis_src_img[j],
                           f'{os.path.basename(src_img_metas[j]["filename"])}')
                subplotimg(axs[0][2], src_gt_semantic_seg[j], f'Source GT',
                           cmap='cityscapes', nc=self.num_classes)
                subplotimg(axs[0][3], src_pseudo_label[j], f'Source PL',
                           cmap='cityscapes', nc=self.num_classes)
                subplotimg(axs[0][4], src_ema_label[j], f'Source EMA PL',
                           cmap='cityscapes', nc=self.num_classes)
                # target visualization
                subplotimg(axs[1][0], vis_tgt_img[j],
                           f'{os.path.basename(tgt_img_metas[j]["filename"])}')
                subplotimg(axs[1][1], vis_tgt_ib_img[j], f'Target IB')
                subplotimg(axs[1][2], tgt_gt_semantic_seg[j], f'Target GT',
                           cmap='cityscapes', nc=self.num_classes)
                subplotimg(axs[1][3], tgt_pseudo_label[j], f'Target PL',
                           cmap='cityscapes', nc=self.num_classes)
                subplotimg(axs[1][4], tgt_ema_ib_label[j], f'Target EMA IB PL',
                           cmap='cityscapes', nc=self.num_classes)
                subplotimg(axs[1][5], tgt_ib_label[j], f'Target IB TEST',
                           cmap='cityscapes', nc=self.num_classes)
                # mixed visualization
                subplotimg(axs[2][0], vis_mix_img[j], f'Mixed')
                subplotimg(axs[2][1], vis_mix_ib_img[j], f'Mixed IB')
                subplotimg(axs[2][2], pseudo_weight[j], 'Pseudo Weight', vmin=0, vmax=1)
                subplotimg(axs[2][3], mixed_lbl[j], f'Mixed PL',
                           cmap='cityscapes', nc=self.num_classes)
                subplotimg(axs[2][4], mix_label_test[j], f'Mixed TEST',
                           cmap='cityscapes', nc=self.num_classes)
                subplotimg(axs[2][5], mix_ib_label_test[j], f'Mixed IB TEST',
                           cmap='cityscapes', nc=self.num_classes)

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
