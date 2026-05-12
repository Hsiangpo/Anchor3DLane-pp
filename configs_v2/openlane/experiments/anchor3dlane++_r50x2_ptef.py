_base_ = './anchor3dlane++_r50x2_pgb_od.py'

model = dict(
    lane_evidence_field=dict(
        tube_enabled=True,
        tube_x_offsets=(-0.5, -0.25, 0.25, 0.5),
        tube_z_offsets=(-0.3, 0.3)))

data = dict(samples_per_gpu=3)
optimizer = dict(type='Adam', lr=1e-5)
lr_config = dict(policy='step', step=[8000], by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=1)

load_from = 'output/openlane/experiments/pgb_od_r50x2/iter_10000.pth'
work_dir = 'output/openlane/experiments/ptef_r50x2'
