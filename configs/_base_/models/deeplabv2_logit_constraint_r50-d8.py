# VBLC dlv2
# model settings

_base_ = ['deeplabv2_r50-d8.py']

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='LogitConstraintLoss', use_sigmoid=False, loss_weight=1.0)))
