# Modifications: Support for soft label in simple_test

from copy import deepcopy

import mmcv
from mmcv.parallel import MMDistributedDataParallel

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models.segmentors import BaseSegmentor
from mmseg.models import build_segmentor


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


class UDADecorator(BaseSegmentor):

    def __init__(self, **cfg):
        super(BaseSegmentor, self).__init__()

        self.model = build_segmentor(deepcopy(cfg['model']))
        self.train_cfg = cfg['model']['train_cfg']
        self.test_cfg = cfg['model']['test_cfg']
        self.num_classes = cfg['model']['decode_head']['num_classes']

    def get_model(self):
        return get_module(self.model)

    def extract_feat(self, img, img_metas=None):
        """Extract features from images."""
        return self.get_model().extract_feat(img, img_metas)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas)

    def forward_train(self, img, img_metas, gt_semantic_seg, return_feat=False):
        """Forward function for training.

        Args:
            img (list[list[Tensor]]): the outer list indicates source/target split
                and the inner list indicates images from different datasets of
                source/target domain, the Tensor within should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[list[dict]]]): the last two lists from outside indicate
                the same as imgs, Inside is a list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (list[list[Tensor]]): Same structure as imgs. Semantic segmentation
                masks used if the architecture supports semantic segmentation task.
                None in list indicates an absence of mask, which could be normal.
            return_feat (Bool): whether to return feature in loss

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        src_img_l, tgt_img_l = img
        src_img_metas_l, tgt_img_metas_l = img_metas
        src_gt_semantic_seg_l = gt_semantic_seg[0]
        losses = dict()
        for i in range(len(src_img_l)):
            src_img = src_img_l[i]
            src_img_metas = src_img_metas_l[i]
            src_gt_semantic_seg = src_gt_semantic_seg_l[i]
            src_losses = self.get_model().forward_train(
                src_img, src_img_metas, src_gt_semantic_seg, return_feat=return_feat)
            losses.update(add_prefix(src_losses, f'src_{src_img_metas[0]["dataset"].replace("Dataset", "").lower()}'))
        return losses

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[List[Tensor | str]]): the outer list indicates different datasets,
                the inner list indicates test-time augmentations, and the inner Tensor
                should have a shape NxCxHxW, which contains all images in the batch.
            img_metas (List[List[List[dict]]]): the outer list indicates different datasets,
                the middle list indicates test-time augs (multiscale, flip, etc.),
                and the inner list indicates images in a batch.
        Returns:
            List[List[data]], data can be empty if no more sample is available
        """
        res = list()
        if isinstance(imgs[0], (list, tuple)):
            for img, meta in zip(imgs, img_metas):
                if meta[0][0]['filename'] is None:
                    res.append([])
                else:
                    result = super(UDADecorator, self).forward_test(img, meta, **kwargs)
                    res.append(result)
            return [res]
        else:
            return super(UDADecorator, self).forward_test(imgs, img_metas, **kwargs)

    def show_result(self,
                    img,
                    result,
                    palette=None,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    opacity=0.5):
        # TODO: fit for MultiTestDataset
        pass

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        return self.get_model().inference(img, img_meta, rescale)

    def simple_test(self, img, img_meta, rescale=True, soft=False):
        """Simple test with single image."""
        return self.get_model().simple_test(img, img_meta, rescale, soft)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)
