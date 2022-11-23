# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA, Mingjia Li. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .utils import get_class_weight, weight_reduce_loss


@LOSSES.register_module()
class LogitConstraintLoss(CrossEntropyLoss):
    """CrossEntropyLoss after Logit Norm.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_lc',
                 avg_non_ignore=False,
                 eps=1e-7):
        super(LogitConstraintLoss, self).__init__(use_sigmoid,
                                                  use_mask,
                                                  reduction,
                                                  class_weight,
                                                  loss_weight,
                                                  loss_name,
                                                  avg_non_ignore)
        self.eps = eps

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        norms = torch.norm(cls_score, p=2, dim=1, keepdim=True) + self.eps
        normed_logit = torch.div(cls_score, norms)
        loss_cls = super(LogitConstraintLoss, self).forward(normed_logit,
                                                            label,
                                                            weight,
                                                            avg_factor,
                                                            reduction_override,
                                                            ignore_index,
                                                            **kwargs)
        return loss_cls
