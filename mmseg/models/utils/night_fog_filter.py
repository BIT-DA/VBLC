# filter adapted from https://github.com/wenyyu/Image-Adaptive-YOLO
# Licensed under the Apache License, Version 2.0
# A copy of the license is available at resources/license_iayolo
# Modifications: Add adaptive param

import kornia.color
import torch
import math


def DarkChannel(im):
    dc, _ = torch.min(im, dim=-1)
    return dc


def AtmLight(im, dark):
    h, w = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort(0)
    indices = indices[(imsz - numpx): imsz]

    atmsum = torch.zeros([1, 3]).cuda()
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def DarkIcA(im, A):
    im3 = torch.empty_like(im)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    return DarkChannel(im3)


def get_saturation(im):
    saturation = (im.max(-1)[0] - im.min(-1)[0]) / (im.max(-1)[0]+1e-10)
    return saturation


def process(im, defog_A, IcA, mode='hsv-s-w4'):
    if mode == 'hsv-s-w4':
        img_s = get_saturation(im)
        s = (-img_s.mean() * 4).exp()
        param = torch.ones_like(img_s) * s
    else:
        raise NotImplementedError(f'{mode} not supported yet!')

    param = param[None, :, :, None]
    tx = 1 - param * IcA

    tx_1 = torch.tile(tx, [1, 1, 1, 3])
    return (im - defog_A[:, None, None, :]) / torch.maximum(tx_1, torch.tensor(0.01)) + defog_A[:, None, None, :]


def blur_filter(X, mode):
    X = X.permute(1, 2, 0).contiguous()

    dark = DarkChannel(X)
    defog_A = AtmLight(X, dark)
    IcA = DarkIcA(X, defog_A)

    IcA = IcA.unsqueeze(-1)

    return process(X, defog_A, IcA, mode=mode)[0].permute(2, 0, 1).contiguous()


def night_fog_filter(normed_img, means, stds, night_map, mode='hsv-s-w4'):
    img = normed_img * stds + means
    img /= 255.
    bs = img.shape[0]
    assert bs == len(night_map)
    for i in range(bs):
        if night_map[i]:
            img[i] = 1 - blur_filter(1 - img[i], mode=mode)
        else:
            img[i] = blur_filter(img[i], mode=mode)
    img *= 255.
    normed_img = (img.float() - means) / stds
    return normed_img
