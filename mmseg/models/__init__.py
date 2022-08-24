# Copyright (c) OpenMMLab. All rights reserved.
# Modifications:
# - Support UDADecorator-based models
# - Support Preprocessors
from .backbones import *  # noqa: F401,F403
from .builder import (BACKBONES, HEADS, LOSSES, SEGMENTORS, UDA, build_backbone,
                      build_head, build_loss, build_segmentor, build_train_model)
from .decode_heads import *  # noqa: F401,F403
from .losses import *  # noqa: F401,F403
from .necks import *  # noqa: F401,F403
from .segmentors import *  # noqa: F401,F403
from .uda import *

__all__ = [
    'BACKBONES', 'HEADS', 'LOSSES', 'SEGMENTORS', 'UDA', 'build_backbone',
    'build_head', 'build_loss', 'build_segmentor', 'build_train_model'
]
