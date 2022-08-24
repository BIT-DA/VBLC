# Copyright (c) OpenMMLab. All rights reserved.
# Modifications:
# - Support UDADataset
from .builder import DATASETS, PIPELINES, build_dataloader, build_dataset
from .cityscapes import CityscapesDataset
from .custom import CustomDataset
from .dataset_wrappers import (ConcatDataset, MultiImageMixDataset,
                               RepeatDataset)

from .acdc import ACDCDataset

from .foggy_rain_cityscapes import FoggyRainCityscapesDataset

from .uda_wrappers import (UDADataset, UDAMultiDataset, MultiTestDataset)

__all__ = [
    'CustomDataset', 'build_dataloader', 'ConcatDataset', 'RepeatDataset',
    'DATASETS', 'build_dataset', 'PIPELINES', 'CityscapesDataset', 'MultiImageMixDataset',
    'UDADataset', 'UDAMultiDataset', 'MultiTestDataset',
    'ACDCDataset', 'FoggyRainCityscapesDataset'
]
