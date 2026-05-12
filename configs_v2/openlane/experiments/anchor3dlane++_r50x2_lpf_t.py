_base_ = './anchor3dlane++_r50x2_frt_eat.py'

model = dict(
    lane_point_field=dict(
        enabled=True,
        apply='final',
        num_heads=4,
        ffn_ratio=2.0,
        dropout=0.0,
        gate_init=-2.1972246,
        use_geometry=True,
        detach_geometry=True),
    accept_calib=dict(
        enabled=True,
        apply='final',
        detach=True,
        freeze_base=True,
        bias_init=4.0,
        hidden_channels=64,
        score_floor=0.75,
        nms_power=1.0))

data = dict(samples_per_gpu=3)
optimizer = dict(type='Adam', lr=2e-5)
lr_config = dict(policy='step', step=[8000], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1)

load_from = 'output/openlane/experiments/frt_eat_r50x2/iter_8000.pth'
work_dir = 'output/openlane/experiments/lpf_t_r50x2'
