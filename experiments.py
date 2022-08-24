# Adapted from https://github.com/lhoyer/DAFormer
import itertools
import logging
import math


def get_model_base(architecture, backbone):
    if 'daformer_' in architecture and 'mitb' in backbone:
        return {
            'mitb5': f'_base_/models/{architecture}_mitb5.py',
            # It's intended that <= mit_b4 refers to mit_b5 config
            'mitb4': f'_base_/models/{architecture}_mitb5.py',
            'mitb3': f'_base_/models/{architecture}_mitb5.py',
            'mitb2': f'_base_/models/{architecture}_mitb5.py',
            'mitb1': f'_base_/models/{architecture}_mitb5.py',
        }[backbone]
    assert 'mit' not in backbone or '-del' in backbone
    return {
        'dlv2': '_base_/models/deeplabv2_r50-d8.py',
        'dlv2lc': '_base_/models/deeplabv2_logit_constraint_r50-d8.py',
    }[architecture]


def get_pretraining_file(backbone):
    if 'mitb5' in backbone:
        return 'pretrained/mit_b5.pth'
    if 'mitb4' in backbone:
        return 'pretrained/mit_b4.pth'
    if 'mitb3' in backbone:
        return 'pretrained/mit_b3.pth'
    if 'mitb2' in backbone:
        return 'pretrained/mit_b2.pth'
    if 'mitb1' in backbone:
        return 'pretrained/mit_b1.pth'
    if 'r101v1c' in backbone:
        return 'open-mmlab://resnet101_v1c'
    return {
        'r50v1c': 'open-mmlab://resnet50_v1c',
    }[backbone]


def get_backbone_cfg(backbone):
    for i in [1, 2, 3, 4, 5]:
        if backbone == f'mitb{i}':
            return dict(type=f'mit_b{i}')
        if backbone == f'mitb{i}-del':
            return dict(_delete_=True, type=f'mit_b{i}')
        if backbone == f'mitb{i}_ae_d8':
            return dict(type=f'mit_b5_ae_d8')
    return {
        'r50v1c': {
            'depth': 50
        },
        'r101v1c': {
            'depth': 101
        },
    }[backbone]


def update_decoder_in_channels(cfg, architecture, backbone):
    cfg.setdefault('model', {}).setdefault('decode_head', {})
    return cfg


def setup_rcs(cfg, temperature):
    cfg.setdefault('data', {}).setdefault('train', {})
    cfg['data']['train']['rare_class_sampling'] = dict(
        min_pixels=3000, class_temp=temperature, min_crop_ratio=0.5)
    return cfg


def generate_experiment_cfgs(id):

    def config_from_vars():
        cfg = {'_base_': ['_base_/default_runtime.py'], 'n_gpus': n_gpus}
        if seed is not None:
            cfg['seed'] = seed

        # Setup model config
        architecture_mod = architecture
        model_base = get_model_base(architecture_mod, backbone)
        cfg['_base_'].append(model_base)
        if pretrained_source is not None:
            cfg['load_from'] = f'pretrained/{pretrained_source}_src_only.pth'
        cfg['model'] = {
            'pretrained': get_pretraining_file(backbone),
            'backbone': get_backbone_cfg(backbone),
        }
        cfg = update_decoder_in_channels(cfg, architecture_mod, backbone)

        # Setup UDA config
        cfg['_base_'].append(
            f'_base_/datasets/uda_{source}_to_{target}_{crop}{suffix}.py')  # unified dataloader
        cfg['_base_'].append(f'_base_/uda/{uda}.py')
        if method_name in uda:
            cfg.setdefault('uda', {})
            cfg['uda']['debug_img_interval'] = debug_img_interval
            cfg['uda']['pseudo_threshold'] = pseudo_threshold
            cfg['uda']['blur'] = blur
            cfg['uda']['color_jitter'] = color_jitter
        cfg['data'] = dict(
            samples_per_gpu=batch_size,
            workers_per_gpu=workers_per_gpu,
            train={})
        if method_name in uda and rcs_T is not None:
            cfg = setup_rcs(cfg, rcs_T)

        # Setup optimizer and schedule
        if method_name in uda:
            cfg['optimizer_config'] = None  # Don't use outer optimizer

        cfg['_base_'].extend(
            [f'_base_/schedules/{opt}.py', f'_base_/schedules/{schedule}.py'])
        cfg['optimizer'] = {'lr': lr}
        cfg['optimizer'].setdefault('paramwise_cfg', {})
        cfg['optimizer']['paramwise_cfg'].setdefault('custom_keys', {})
        opt_param_cfg = cfg['optimizer']['paramwise_cfg']['custom_keys']
        if pmult:
            opt_param_cfg['head'] = dict(lr_mult=10.)
        if 'mit' in backbone:
            opt_param_cfg['pos_block'] = dict(decay_mult=0.)
            opt_param_cfg['norm'] = dict(decay_mult=0.)

        # Setup runner
        cfg['runner'] = dict(type='IterBasedRunner', max_iters=iters)
        cfg['checkpoint_config'] = dict(
            by_epoch=False, interval=iters, max_keep_ckpts=1)
        cfg['evaluation'] = dict(interval=1000, metric='mIoU')
        cfg['log_config'] = dict(interval=50)

        # Construct config name
        uda_mod = uda
        if method_name in uda and rcs_T is not None:
            uda_mod += f'_rcs{rcs_T}'

        cfg['name'] = f'{source}2{target}_{uda_mod}_{architecture_mod}_' \
                      f'{backbone}_{schedule}'
        cfg['exp'] = id
        cfg['name_dataset'] = f'{source}2{target}'
        cfg['name_architecture'] = f'{architecture_mod}_{backbone}'
        cfg['name_encoder'] = backbone
        cfg['name_decoder'] = architecture_mod
        cfg['name_uda'] = uda_mod
        cfg['name_opt'] = f'{opt}_{lr}_pm{pmult}_{schedule}' \
                          f'_{n_gpus}x{batch_size}_{iters // 1000}k'
        if seed is not None:
            cfg['name'] += f'_s{seed}'
        cfg['name'] = cfg['name'].replace('.', '').replace('True', 'T') \
            .replace('False', 'F').replace('cityscapes', 'city')
        return cfg

    # -------------------------------------------------------------------------
    # Set some defaults
    # -------------------------------------------------------------------------
    cfgs = []
    method_name = 'acdc'
    n_gpus = 1
    batch_size = 2
    iters = 40000
    debug_img_interval = 1000
    pretrained_source = None
    opt, lr, schedule, pmult = 'adamw', 0.00006, 'poly10warm', True
    crop = '512x512'
    suffix = ''
    datasets = [
        ('gta', 'cityscapes'),
    ]
    architecture = None
    workers_per_gpu = 2  # avoid failure in training
    rcs_T = None
    pseudo_threshold = 0.968
    blur = False
    color_jitter = False
    # -------------------------------------------------------------------------
    # Architecture Startup Test
    # -------------------------------------------------------------------------
    if id == 0:
        iters = 2
        seeds = [0]
        models = [
            ('dlv2lc', 'r101v1c'),
            ('daformer_sepaspp_logit_constraint', 'mitb5'),
        ]
        datasets = [
            ('cityscapes', 'acdc'),
            ('cityscapes', 'foggyraincityscapes'),
        ]
        crop = '640x640'
        udas = ['test_uda']
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            # cfg['log_level'] = logging.ERROR
            cfg['evaluation']['interval'] = 1
            cfg['log_config']['interval'] = 1
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Cityscapes --> ACDC (SegFormer MiT-B5)
    # -------------------------------------------------------------------------
    elif id == 1:
        seeds = [0]
        models = [
            ('daformer_sepaspp_logit_constraint', 'mitb5'),
        ]
        datasets = [
            ('cityscapes', 'acdc')
        ]
        crop = '640x640'
        udas = ['acdc_dacs_ema_night_fog_saturation_w4']
        pseudo_threshold = 0.9
        rcs_T = 0.01
        blur = True
        color_jitter = True
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Cityscapes --> ACDC (ResNet-101)
    # -------------------------------------------------------------------------
    elif id == 2:
        seeds = [0]
        models = [
            ('dlv2lc', 'r101v1c'),
        ]
        datasets = [
            ('cityscapes', 'acdc')
        ]
        crop = '640x640'
        udas = ['acdc_dacs_ema_night_fog_saturation_w4']
        pseudo_threshold = 0.9
        rcs_T = 0.01
        blur = True
        color_jitter = True
        iters = 100000
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Cityscapes --> FoggyCityscapes + RainCityscapes (SegFormer MiT-B5)
    # -------------------------------------------------------------------------
    elif id == 3:
        seeds = [0]
        models = [
            ('daformer_sepaspp_logit_constraint', 'mitb5'),
        ]
        datasets = [
            ('cityscapes', 'foggyraincityscapes')
        ]
        crop = '640x640'
        udas = ['acdc_dacs_ema_night_fog_saturation_w4']
        pseudo_threshold = 0.9
        rcs_T = 0.01
        blur = True
        color_jitter = True
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Cityscapes --> FoggyCityscapes + RainCityscapes (ResNet-101)
    # -------------------------------------------------------------------------
    elif id == 4:
        seeds = [0]
        models = [
            ('dlv2lc', 'r101v1c'),
        ]
        datasets = [
            ('cityscapes', 'foggyraincityscapes')
        ]
        crop = '640x640'
        udas = ['acdc_dacs_ema_night_fog_saturation_w4']
        pseudo_threshold = 0.9
        rcs_T = 0.01
        blur = True
        color_jitter = True
        for (source, target), (architecture, backbone), uda, seed in \
                itertools.product(datasets, models, udas, seeds):
            cfg = config_from_vars()
            cfgs.append(cfg)
    else:
        raise NotImplementedError('Unknown id {}'.format(id))

    return cfgs
