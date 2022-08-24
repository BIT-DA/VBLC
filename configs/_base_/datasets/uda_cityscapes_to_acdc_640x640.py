# dataset settings
dataset_type = 'ACDCDataset'
data_root = 'data/acdc/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 640)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
acdc_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1280, 720)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',
        source=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline),
        target=dict(
            type='ACDCDataset',
            data_root='data/acdc/',
            img_dir=[
                'rgb_anon/fog/train',
                'rgb_anon/night/train',
                'rgb_anon/rain/train',
                'rgb_anon/snow/train',
            ],
            ann_dir=[
                'gt/fog/train',
                'gt/night/train',
                'gt/rain/train',
                'gt/snow/train',
            ],
            pipeline=acdc_train_pipeline)),
    val=dict(
        type='ACDCDataset',
        data_root='data/acdc/',
        img_dir=[
            'rgb_anon/fog/val',
            'rgb_anon/night/val',
            'rgb_anon/rain/val',
            'rgb_anon/snow/val',
        ],
        ann_dir=[
            'gt/fog/val',
            'gt/night/val',
            'gt/rain/val',
            'gt/snow/val',
        ],
        separate_eval=False,
        pipeline=test_pipeline),
    test=dict(
        type='ACDCDataset',
        data_root='data/acdc/',
        img_dir=[
            'rgb_anon/fog/test',
            'rgb_anon/night/test',
            'rgb_anon/rain/test',
            'rgb_anon/snow/test',
        ],
        separate_eval=False,
        test_mode=True,
        pipeline=test_pipeline)
)
